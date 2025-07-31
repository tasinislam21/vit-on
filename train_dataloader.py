import torch.utils.data as data
import os
import torch
import os.path as osp

class BaseDataset(data.Dataset):
    def __init__(self, padding = 64):
        super(BaseDataset, self).__init__()
        self.name_list = os.listdir('dataset_binary/gt')
        self.m = torch.nn.ZeroPad2d((padding, padding, 0, 0))

    def __getitem__(self, index):
        name = self.name_list[index]
        gt = self.m(torch.load(osp.join('dataset_binary', 'gt', name)))
        input_person = self.m(torch.load(osp.join('dataset_binary', 'agnostic', name)))
        input_skeleton = self.m(torch.load(osp.join('dataset_binary', 'input_skeleton', name)))
        input_clothing = self.m(torch.load(osp.join('dataset_binary', 'input_clothing', name)))
        warp_clothing = self.m(torch.load(osp.join('dataset_binary', 'warped_cloth', name)))
        clip_clothing = torch.load(osp.join('dataset_binary', 'clip_clothing', name))

        return {'gt': gt, 'input_person': input_person,
                'input_skeleton': input_skeleton,
                'input_clothing': input_clothing,
                'clip_clothing': clip_clothing,
                'warp_clothing': warp_clothing,
                'name': name}

    def __len__(self):
        return len(self.name_list)