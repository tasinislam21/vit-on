from transformers import CLIPVisionModel, CLIPProcessor
import torch.utils.data as data
import os
import os.path as osp
from PIL import Image
import torch
import torchvision.transforms as transforms


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

mean_skeleton = [0.05440789, 0.07170792, 0.04121648]
std_skeleton = [0.20046051, 0.23692659, 0.16482468]

clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
clip_encoder.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.name_list = os.listdir('dataset_test/image')
        self.transform_clothes = get_transform(mean=mean_clothing, std=std_clothing)
        self.transform_candidate = get_transform(mean=mean_candidate, std=std_candidate)
        self.transform_skeleton = get_transform(mean=mean_skeleton, std=std_skeleton)
        self.m = torch.nn.ZeroPad2d((64, 64, 0, 0))
        self.dataset_path = 'dataset_test'
        self.human_names = []
        self.cloth_names = []
        with open('dataset_test/pair.txt', 'r') as f:
            for line in f.readlines():
                h_name, c_name = line.strip().split()
                self.human_names.append(h_name)
                self.cloth_names.append(c_name)

    def __getitem__(self, index):
        h_name = self.human_names[index]
        c_name = self.cloth_names[index]

        S_path = osp.join(self.dataset_path, 'image-densepose', h_name)
        skeleton = Image.open(S_path).convert('RGB')

        C_path = osp.join(self.dataset_path, 'cloth', c_name)
        color = Image.open(C_path).convert('RGB')

        D_path = osp.join(self.dataset_path, 'agnostic-v3.2', h_name)
        agnostic_img = Image.open(D_path).convert('RGB')

        clothing = self.m(self.transform_clothes(color))
        skeleton = self.m(self.transform_skeleton(skeleton))
        agnostic = self.m(self.transform_candidate(agnostic_img))

        clip_clothing = clip_processor(images=color, return_tensors="pt")
        clip_clothing = {k: v for k, v in clip_clothing.items()}
        clip_clothing = clip_encoder(**clip_clothing).last_hidden_state

        return {'name':h_name,
                'input_person': agnostic,
                'input_skeleton': skeleton,
                'input_clothing': clothing,
                'clip_clothing': clip_clothing
                }

    def __len__(self):
        return len(self.name_list)
