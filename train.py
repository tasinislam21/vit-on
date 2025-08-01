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
from models import DiT
from train_dataloader import BaseDataset
import torchvision
import kornia.augmentation as K
import os
import os.path as osp

mean_candidate = [0.74112587, 0.69617281, 0.68865463]
std_candidate = [0.2941623, 0.30806473, 0.30613222]

augment = K.RandomAffine(degrees=15, translate=(0.1, 0.1), p=1.0)

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
    checkpoint = None
    if (args.checkpoint_path != None) and args.current_epoch != 0:
        checkpoint = torch.load(os.path.join(args.checkpoint_path, "backup_{}.pt".format(args.current_epoch)), weights_only=False, map_location='cpu')
        print("loaded checkpoint!")
    model = DiT(input_size=args.latent_size, depth=args.model_depth)
    ema = deepcopy(model)  # Create an EMA of the model for use after training
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        print("model weight restored!")
    model.to(device)
    ema.to(device)
    requires_grad(ema, False)
    model = DDP(model, device_ids=[rank])
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
    ).to(device)
    vae.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)
    if checkpoint is not None:
        opt.load_state_dict(checkpoint['opt'])
        print("optimiser restored!")

    train_dataset = BaseDataset(padding=32)
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
    if checkpoint is None:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()
    ema.eval()
    train_steps = 0

    if get_rank() == 0:
        writer = SummaryWriter('runs')

    def get_loss(input_clothing, input_pose, gt_warp, warp_clip):
        b, _, _, _ = input_clothing.shape
        timesteps = torch.randint(0, T, (b,), device=device)
        timesteps = timesteps.long()
        #x_noisy, noise = forward_diffusion_sample(gt, timesteps)
        x_noisy, noise = forward_diffusion_sample(gt_warp, timesteps)
        input_pose = torch.cat([input_pose, x_noisy], dim=1)
        noise_pred, warp_pred = model(input_clothing, input_pose, timesteps.float())
        noise_loss = mseloss(noise_pred, noise)
        warp_loss = mseloss(warp_pred, warp_clip)
        return noise_loss, warp_loss

    @torch.no_grad()
    def sample_timestep(input_pose, input_clothing, t):
        betas_t = get_index_from_list(betas, t, input_pose[:,4:8].shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, input_pose[:,4:8].shape
        )
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, input_pose[:,4:8].shape)
        # Call model (current image - noise prediction)
        with torch.cuda.amp.autocast():
            sample_output, _ = ema(input_clothing, input_pose, t.float())
        model_mean = sqrt_recip_alphas_t * (
                input_pose[:,4:8] - betas_t * sample_output / sqrt_one_minus_alphas_cumprod_t
        )
        if t.item() == 0:
            return model_mean
        else:
            noise = torch.randn_like(input_pose[:,4:8])
            posterior_variance_t = get_index_from_list(posterior_variance, t, input_pose[:,4:8].shape)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def VAE_decode(latent):
        latent = 1 / 0.18215 * latent
        latent = vae.decode(latent).sample
        latent = inv_normalize(latent).clamp(0, 1)
        return latent

    for epoch in tqdm.tqdm(range(args.current_epoch, 1000)):
        for data in train_dataloader:
            input_skeleton = data['input_skeleton'].to(device)
            input_clothing = data['input_clothing'].to(device)
            warp_clip = data['clip_clothing'].to(device)
            gt_warp = data['warp_clothing'].to(device)
            encoded_skeleton = vae.encode(input_skeleton).latent_dist.sample() * 0.18215
            encoded_clothing = vae.encode(input_clothing).latent_dist.sample() * 0.18215
            encoded_clothing = augment(encoded_clothing)
            encoded_warp = vae.encode(gt_warp).latent_dist.sample() * 0.18215

            loss_noise, loss_warp = get_loss(input_clothing=encoded_clothing, input_pose=encoded_skeleton, gt_warp=encoded_warp, warp_clip=warp_clip)

            loss = loss_noise + loss_warp
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)
            if train_steps % 1000 == 0 and get_rank() == 0:
                writer.add_scalar('noise', loss_noise, train_steps)
                writer.add_scalar('warp', loss_warp, train_steps)
            train_steps += 1
        if get_rank() == 0 and epoch % 20 == 0:
            checkpoint = { # Make checkpoint
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args
            }
            if not os.path.exists(osp.join('checkpoint')):
                os.makedirs(osp.join('checkpoint'))

            torch.save(checkpoint, 'checkpoint/backup_{}.pt'.format(str(epoch)))

            b, _, _, _ = encoded_skeleton.shape  # Also save some image samples
            noise = torch.randn([b, 4, args.latent_size, args.latent_size]).to(device)
            pose_data = torch.cat([encoded_skeleton, noise], dim=1)
            for i in range(0, T)[::-1]:
                t = torch.full((1,), i, device=device).long()
                noise = sample_timestep(pose_data, encoded_clothing, t)
                pose_data[:, 4:8] = noise
            final_image = VAE_decode(pose_data[:, 4:8])
            writer.add_image('Fused', torchvision.utils.make_grid(inv_normalize(final_image)), train_steps)
            writer.add_image('Warp', torchvision.utils.make_grid(inv_normalize(gt_warp)), train_steps)
            writer.add_image('Clothing', torchvision.utils.make_grid(inv_normalize(input_clothing)), train_steps)

    torch.save(ema.state_dict(), 'checkpoint/final.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--model-depth", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--current-epoch", type=int, default=0)

    args = parser.parse_args()
    main(args)