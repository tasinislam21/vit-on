import torch
import numpy as np
import tqdm
import model_v1
from test_dataloader import BaseDataset
from diffusers import AutoencoderKL
import torch.nn.functional as F
from PIL import Image
import itertools
import torchvision.transforms as transforms

device = 'cuda'

mean_candidate = [0.5, 0.5, 0.5]
std_candidate = [0.5, 0.5, 0.5]

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_candidate, std_candidate)],
    std=[1/s for s in std_candidate]
)

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

checkpoint = torch.load("ema_490.pt", map_location='cpu')
model = model_v1.DiT().to(device)
model.eval()
model.load_state_dict(checkpoint)
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="vae",
    revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
).to(device)
vae.requires_grad_(False)
vae_trainable_params = []
for name, param in vae.named_parameters():
    if 'decoder' in name:
        param.requires_grad = True
        vae_trainable_params.append(param)

params_to_optimize = itertools.chain(vae_trainable_params)
optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=5e-5,
    betas=(0.9, 0.99),
    weight_decay=1e-2,
    eps=1e-08)

train_dataset = BaseDataset()

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=0,
    drop_last=True)


@torch.no_grad()
def sample_timestep(input_person, input_clothing, t):
    betas_t = get_index_from_list(betas, t, input_person[:,8:12].shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, input_person[:,8:12].shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, input_person[:,8:12].shape)
    # Call model (current image - noise prediction)
    with torch.cuda.amp.autocast():
        sample_output = model(input_person, input_clothing, t.float())
    model_mean = sqrt_recip_alphas_t * (
            input_person[:,8:12] - betas_t * sample_output / sqrt_one_minus_alphas_cumprod_t
    )
    if t.item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(input_person[:,8:12])
        posterior_variance_t = get_index_from_list(posterior_variance, t, input_person[:,8:12].shape)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def tensor2image(tensor):
    tensor = (tensor[0].clone()) * 255
    tensor = tensor.cpu().clamp(0, 255)
    tensor = tensor.detach().numpy().astype('uint8')
    tensor = tensor.swapaxes(0, 1).swapaxes(1, 2)
    image_pil = Image.fromarray(tensor)
    return image_pil

@torch.no_grad()
def VAE_decode(latent):
    latent = 1 / 0.18215 * latent
    latent = vae.decode(latent).sample
    latent = inv_normalize(latent).clamp(0, 1)
    return latent

for data in train_dataloader:
    input_person = data['input_person'].to(device)
    input_skeleton = data['input_skeleton'].to(device)
    input_clothing = data['input_clothing'].to(device)
    for vae_step in tqdm.tqdm(range(10)):
        vae.train()
        latents = vae.encode(input_person).latent_dist.sample()
        latents_c = vae.encode(input_clothing).latent_dist.sample()
        latents *= 0.18215
        latents_c *= 0.18215
        latents = 1 / 0.18215 * latents
        latents_c = 1 / 0.18215 * latents_c
        pred_images = vae.decode(latents).sample
        pred_c = vae.decode(latents_c).sample
        pred_images = pred_images.clamp(-1, 1)
        pred_c = pred_c.clamp(-1, 1)
        loss = F.mse_loss(pred_images.float(), input_person.clamp(-1, 1).float(), reduction="mean")
        loss += F.mse_loss(pred_c.float(), input_clothing.clamp(-1, 1).float(), reduction="mean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    vae.eval()
    encoded_person = vae.encode(input_person).latent_dist.sample() * 0.18215
    encoded_skeleton = vae.encode(input_skeleton).latent_dist.sample() * 0.18215
    encoded_clothing = vae.encode(input_clothing).latent_dist.sample() * 0.18215
    noise = torch.randn([1, 4, 64, 64]).to(device)
    person_data = torch.cat([encoded_person, encoded_skeleton, noise], dim=1)
    for i in tqdm.tqdm(range(0, T)[::-1]):
        t = torch.full((1,), i, device=device).long()
        noise = sample_timestep(person_data, encoded_clothing, t)
        person_data[:, 8:12] = noise
    final_image = VAE_decode(person_data[:, 8:12])
    jpg = tensor2image(final_image)
    jpg.save('result/{}.jpg'.format(data['name'].item()))