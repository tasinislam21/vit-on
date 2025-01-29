import argparse
import torch
import numpy as np
from diffusers import AutoencoderKL
import torch.nn.functional as F
from torch.distributed import get_rank
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from copy import deepcopy
from collections import OrderedDict
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import model_v2
from train_dataloader import BaseDataset

mean_candidate = [0.74112587, 0.69617281, 0.68865463]
std_candidate = [0.2941623, 0.30806473, 0.30613222]

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_candidate, std_candidate)],
    std=[1/s for s in std_candidate]
)

def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def cleanup():
    dist.destroy_process_group()

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cosine_beta_schedule(timesteps, start=0.0001, end=0.02):
    betas = []
    for i in reversed(range(timesteps)):
        T = timesteps - 1
        beta = start + 0.5 * (end - start) * (1 + np.cos((i / T) * np.pi))
        betas.append(beta)
    return torch.Tensor(betas)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(t.device) * x_0.to(t.device) \
    + sqrt_one_minus_alphas_cumprod_t.to(t.device) * noise.to(t.device), noise.to(t.device)

T = 100
betas = cosine_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

mseloss = torch.nn.MSELoss()

def main(args):
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    checkpoint = torch.load("checkpoint/backup_380.pt", map_location='cpu')
    model = model_v2.DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16).to(device)
    model.x_embedder.proj = torch.nn.Conv2d(in_channels=16, out_channels=1152, kernel_size=2, stride=2).to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    model.load_state_dict(checkpoint['model'])
    ema.load_state_dict(checkpoint['ema'])
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
    ).to(device)
    vae.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    opt.load_state_dict(checkpoint['opt'])
    train_dataset = BaseDataset()
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.global_batch_size,
        sampler=sampler,
        num_workers=4,
        drop_last=False)

    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()
    train_steps = 0

    if get_rank() == 0:
        writer = SummaryWriter('runs')

    def get_loss(input_person, gt):
        b, _, _, _ = input_person.shape
        timesteps = torch.randint(0, T, (b,), device=device)
        timesteps = timesteps.long()
        x_noisy, noise = forward_diffusion_sample(gt, timesteps)
        input_person = torch.cat([input_person, x_noisy], dim=1)
        noise_pred = model(input_person, timesteps.float())
        loss = mseloss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample_timestep(input_person, t):
        betas_t = get_index_from_list(betas, t, input_person[:,12:16].shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, input_person[:,12:16].shape
        )
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, input_person[:,12:16].shape)
        # Call model (current image - noise prediction)
        with torch.cuda.amp.autocast():
            sample_output = ema(input_person, t.float())
        model_mean = sqrt_recip_alphas_t * (
                input_person[:,12:16] - betas_t * sample_output / sqrt_one_minus_alphas_cumprod_t
        )
        if t.item() == 0:
            return model_mean
        else:
            noise = torch.randn_like(input_person[:,12:16])
            posterior_variance_t = get_index_from_list(posterior_variance, t, input_person[:,12:16].shape)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def VAE_decode(latent):
        latent = 1 / 0.18215 * latent
        latent = vae.decode(latent).sample
        latent = inv_normalize(latent).clamp(0, 1)
        return latent

    for epoch in range(381, 1000):
        for data in train_dataloader:
            input_person = data['input_person'].to(device)
            input_skeleton = data['input_skeleton'].to(device)
            input_clothing = data['input_clothing'].to(device)
            gt = data['gt'].to(device)
            encoded_person = vae.encode(input_person).latent_dist.sample() * 0.18215
            encoded_skeleton = vae.encode(input_skeleton).latent_dist.sample() * 0.18215
            encoded_clothing = vae.encode(input_clothing).latent_dist.sample() * 0.18215
            encoded_gt = vae.encode(gt).latent_dist.sample() * 0.18215
            person_data = torch.cat([encoded_person, encoded_skeleton, encoded_clothing], dim=1)
            loss = get_loss(input_person=person_data, gt=encoded_gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)
            if train_steps % 100 == 0 and get_rank() == 0:
                writer.add_scalar('loss', loss, train_steps)
            if train_steps % 500 == 0 and get_rank() == 0:
                encoded_person = encoded_person[0].unsqueeze(0)
                encoded_skeleton = encoded_skeleton[0].unsqueeze(0)
                encoded_clothing = encoded_clothing[0].unsqueeze(0)
                noise = torch.randn([1, 4, 32, 32]).to(device)
                person_data = torch.cat([encoded_person, encoded_skeleton, encoded_clothing, noise], dim=1)
                for i in range(0, T)[::-1]:
                    t = torch.full((1,), i, device=device).long()
                    noise = sample_timestep(person_data, t)
                    person_data[:,12:16] = noise
                final_image = VAE_decode(person_data[:,12:16])
                writer.add_image('Person', input_person[0], train_steps)
                writer.add_image('Clothing', input_clothing[0], train_steps)
                writer.add_image('Fused', final_image[0], train_steps)
            train_steps += 1
        if get_rank() == 0 and epoch % 10 == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args
            }
            torch.save(checkpoint, 'checkpoint/backup_{}.pt'.format(str(epoch)))
    torch.save(ema.state_dict(), 'checkpoint/final.pt')


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
