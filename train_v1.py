import argparse
import torch
import numpy as np
import tqdm
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
import model_v1
from train_dataloader import BaseDataset
import torchvision
from transformers import CLIPVisionModel

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
    #checkpoints = torch.load("checkpoint/backup_70.pt", map_location="cpu")
    model = model_v1.DiT(input_size=args.latent_size, depth=8).to(device)
    #model.load_state_dict(checkpoints["model"])
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    #ema.load_state_dict(checkpoints["ema"])
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])

    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
    ).to(device)
    vae.requires_grad_(False)

    clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_encoder.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    #opt.load_state_dict(checkpoints["opt"])
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

    def get_loss(input_person, input_clothing, clip_clothing, gt):
        b, _, _, _ = input_person.shape
        timesteps = torch.randint(0, T, (b,), device=device)
        timesteps = timesteps.long()
        x_noisy, noise = forward_diffusion_sample(gt, timesteps)
        input_person = torch.cat([input_person, x_noisy], dim=1)

        noise_pred = model(input_person, input_clothing, clip_clothing, timesteps.float())
        loss = mseloss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample_timestep(input_person, input_clothing, clip_clothing, t):
        betas_t = get_index_from_list(betas, t, input_person[:,8:12].shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, input_person[:,8:12].shape
        )
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, input_person[:,8:12].shape)
        # Call model (current image - noise prediction)
        with torch.cuda.amp.autocast():
            sample_output = ema(input_person, input_clothing, clip_clothing, t.float())
        model_mean = sqrt_recip_alphas_t * (
                input_person[:,8:12] - betas_t * sample_output / sqrt_one_minus_alphas_cumprod_t
        )
        if t.item() == 0:
            return model_mean
        else:
            noise = torch.randn_like(input_person[:,8:12])
            posterior_variance_t = get_index_from_list(posterior_variance, t, input_person[:,8:12].shape)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def VAE_decode(latent):
        latent = 1 / 0.18215 * latent
        latent = vae.decode(latent).sample
        latent = inv_normalize(latent).clamp(0, 1)
        return latent

    for epoch in tqdm.tqdm(range(1000)):
        for data in train_dataloader:
            input_person = data['input_person'].to(device)
            input_skeleton = data['input_skeleton'].to(device)
            input_clothing = data['input_clothing'].to(device)

            gt = data['gt'].to(device)
            encoded_person = vae.encode(input_person)
            encoded_skeleton = vae.encode(input_skeleton)
            encoded_clothing = vae.encode(input_clothing)
            encoded_gt = vae.encode(gt)
            clip_clothing = clip_encoder(**input_clothing).last_hidden_state.to(device)

            person_data = torch.cat([encoded_person, encoded_skeleton], dim=1)
            clothing_data = torch.cat([encoded_clothing, encoded_skeleton], dim=1)
            loss = get_loss(input_person=person_data, input_clothing=clothing_data, clip_clothing=clip_clothing, gt=encoded_gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)
            if train_steps % 1000 == 0 and get_rank() == 0:
                writer.add_scalar('loss', loss, train_steps)
            if train_steps % 4000 == 0 and get_rank() == 0:
                b, _, _, _ = encoded_person.shape
                noise = torch.randn([b, 4, args.latent_size, args.latent_size]).to(device)
                person_data = torch.cat([encoded_person, encoded_skeleton, noise], dim=1)
                for i in range(0, T)[::-1]:
                    t = torch.full((1,), i, device=device).long()
                    noise = sample_timestep(person_data, clothing_data, clip_clothing, t)
                    person_data[:,8:12] = noise
                final_image = VAE_decode(person_data[:,8:12])
                writer.add_image('Person', torchvision.utils.make_grid(inv_normalize(input_person)), train_steps)
                writer.add_image('Clothing', torchvision.utils.make_grid(inv_normalize(input_clothing)), train_steps)
                writer.add_image('Fused', torchvision.utils.make_grid(inv_normalize(final_image)), train_steps)
            train_steps += 1
        if get_rank() == 0 and epoch % 20 == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args
            }
            torch.save(checkpoint, 'checkpoint/backup_{}.pt'.format(str(epoch)))
    torch.save(ema.state_dict(), 'checkpoint/final.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--latent-size", type=int, default=64)
    args = parser.parse_args()
    main(args)