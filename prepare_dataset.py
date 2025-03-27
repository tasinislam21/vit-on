import os.path as osp
import torchvision.transforms as transforms
from PIL import Image
import torch
import tqdm
import os
from transformers import CLIPVisionModel, CLIPProcessor


def get_transform(normalize=True, mean=None, std=None, downsize=True):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if downsize:
        transform_list += [transforms.Resize((256, 192))]
    if normalize:
        transform_list += [transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(transform_list)

mean_clothing = [0.73949153, 0.70635068, 0.71736564]
std_clothing = [0.34867646, 0.36374153, 0.35065262]

mean_candidate = [0.74112587, 0.69617281, 0.68865463]
std_candidate = [0.2941623, 0.30806473, 0.30613222]

mean_skeleton = [0.5, 0.5, 0.5]
std_skeleton = [0.5, 0.5, 0.5]

transform_mask = get_transform(normalize=False)
transform_clothes = get_transform(mean=mean_clothing, std=std_clothing)
transform_candidate = get_transform(mean=mean_candidate, std=std_candidate)
transform_skeleton = get_transform(mean=mean_skeleton, std=std_skeleton)

clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
clip_encoder.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

name_list = os.listdir('dataset/image')

for i in tqdm.tqdm(range(len(name_list))):
    file_name = name_list[i]
    B_path = osp.join('dataset', 'image', file_name)
    image = Image.open(B_path).convert('RGB')

    S_path = osp.join('dataset', 'image-densepose', file_name)
    skeleton = Image.open(S_path).convert('RGB')

    C_path = osp.join('dataset', 'cloth', file_name)
    color = Image.open(C_path).convert('RGB')

    D_path = osp.join('dataset', 'agnostic-v3.2', file_name)
    agnostic = Image.open(D_path).convert('RGB')

    ground_truth = transform_candidate(image)
    clothing = transform_clothes(color)
    skeleton = transform_skeleton(skeleton)
    agnostic = transform_candidate(agnostic)

    if not os.path.exists(osp.join('dataset_binary','gt')):
        os.makedirs(osp.join('dataset_binary','gt'))
    if not os.path.exists(osp.join('dataset_binary','input_skeleton')):
        os.makedirs(osp.join('dataset_binary','input_skeleton'))
    if not os.path.exists(osp.join('dataset_binary','input_clothing')):
        os.makedirs(osp.join('dataset_binary','input_clothing'))
    if not os.path.exists(osp.join('dataset_binary','agnostic')):
        os.makedirs(osp.join('dataset_binary','agnostic'))
    if not os.path.exists(osp.join('dataset_binary','clip_clothing')):
        os.makedirs(osp.join('dataset_binary','clip_clothing'))

    clip_clothing = clip_processor(images=list(color), return_tensors="pt")
    clip_clothing = {k: v for k, v in clip_clothing.items()}
    clip_clothing = clip_encoder(**clip_clothing).last_hidden_state

    torch.save(ground_truth, osp.join('dataset_binary','gt', file_name.replace('.jpg', '.pt')))
    torch.save(skeleton, osp.join('dataset_binary','input_skeleton', file_name.replace('.jpg', '.pt')))
    torch.save(clothing, osp.join('dataset_binary','input_clothing', file_name.replace('.jpg', '.pt')))
    torch.save(agnostic, osp.join('dataset_binary','agnostic', file_name.replace('.jpg', '.pt')))
    torch.save(clip_clothing, osp.join('dataset_binary','clip_clothing', file_name.replace('.jpg', '.pt')))