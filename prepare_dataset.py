import os.path as osp
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
import os
from transformers import CLIPVisionModel, CLIPProcessor
from multiprocessing import Process
import argparse

def get_transform(normalize=True, mean=None, std=None):
    transform_list = []
    transform_list += [transforms.Resize((512, 384))]
    transform_list += [transforms.ToTensor()]
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

def f(image_list):
    for file_name in image_list:
        B_path = osp.join('dataset', 'image', file_name)
        image = Image.open(B_path).convert('RGB')

        X_path = osp.join('dataset', 'image-parse-v3', file_name.replace('jpg', 'png'))
        person = Image.open(B_path).resize((384, 512))
        segment = Image.open(X_path).convert('L').resize((384, 512), resample=0)
        segment_np = np.array(segment)
        person_np = np.array(person)
        person_np = np.transpose(person_np, (2, 0, 1))
        mask = (segment_np == 126).astype(int)
        mask = np.expand_dims(mask, axis=0)
        extracted_cloth_np = mask * person_np
        extracted_cloth = Image.fromarray(np.transpose(extracted_cloth_np, (1, 2, 0)).astype('uint8'), 'RGB')

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
        extracted_cloth = transform_candidate(extracted_cloth)

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
        if not os.path.exists(osp.join('dataset_binary','warped_cloth')):
            os.makedirs(osp.join('dataset_binary','warped_cloth'))

        clip_clothing = clip_processor(images=color, return_tensors="pt")
        clip_clothing = {k: v for k, v in clip_clothing.items()}
        clip_clothing = clip_encoder(**clip_clothing).last_hidden_state

        torch.save(ground_truth, osp.join('dataset_binary','gt', file_name.replace('.jpg', '.pt')))
        torch.save(extracted_cloth, osp.join('dataset_binary','warped_cloth', file_name.replace('.jpg', '.pt')))
        torch.save(skeleton, osp.join('dataset_binary','input_skeleton', file_name.replace('.jpg', '.pt')))
        torch.save(clothing, osp.join('dataset_binary','input_clothing', file_name.replace('.jpg', '.pt')))
        torch.save(agnostic, osp.join('dataset_binary','agnostic', file_name.replace('.jpg', '.pt')))
        torch.save(clip_clothing[0], osp.join('dataset_binary','clip_clothing', file_name.replace('.jpg', '.pt')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-core-use", type=int, default=8)
    args = parser.parse_args()
    name_list = os.listdir('dataset/image')
    smaller_list = np.array_split(name_list, args.cpu_core_use)
    for smaller in smaller_list:
        p = Process(target=f, args=(smaller,))
        p.start()
        p.join()