from model import DiT
import torch

device = 'cuda'
model = DiT(input_size=64, depth=1).to(device)

data = torch.randn([4,16,64,64]).to(device)
clip = torch.randn([4,50,768]).to(device)
t = torch.full((4,), 5, device=device).long()

output = model(data, clip, t.float())