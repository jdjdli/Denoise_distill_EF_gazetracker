from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
from torchvision import transforms
import copy
import torch.nn.functional as F


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, frames_path: list, voxels_path: list, template_frames_path_1:list, template_voxels_path_1:list, template_frames_path_2:list, template_voxels_path_2:list, template_frames_path_3:list, template_voxels_path_3:list, template_frames_path_4:list, template_voxels_path_4:list, template_frames_path_5:list, template_voxels_path_5:list,
                 position_label_path: list, position_weight_label_path: list, transform=None):
        self.frames_path = frames_path
        self.voxels_path = voxels_path

        self.template_frames_path_1 = template_frames_path_1
        self.template_voxels_path_1 = template_voxels_path_1

        self.template_frames_path_2 = template_frames_path_2
        self.template_voxels_path_2 = template_voxels_path_2

        self.template_frames_path_3 = template_frames_path_3
        self.template_voxels_path_3 = template_voxels_path_3

        self.template_frames_path_4 = template_frames_path_4
        self.template_voxels_path_4 = template_voxels_path_4

        self.template_frames_path_5 = template_frames_path_5
        self.template_voxels_path_5 = template_voxels_path_5

        self.position_label_path = position_label_path

        self.position_weight_label_path = position_weight_label_path

        self.transform = transform

    def __len__(self):
        return len(self.frames_path)

    def __getitem__(self, item):
        img = Image.open(self.frames_path[item])
        img_rgb = img.convert('RGB')
        # RGB为彩色图片，L为灰度图片
        if img_rgb.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.frames_path[item]))
        
        # tmp_1
        img_tmp_1 = Image.open(self.template_frames_path_1[item])
        img_rgb_tmp_1 = img_tmp_1.convert('RGB')
        # RGB为彩色图片，L为灰度图片
        if img_rgb_tmp_1.mode != 'RGB':
            raise ValueError("template image: {} isn't RGB mode.".format(self.template_frames_path_1[item]))


        # tmp_2
        img_tmp_2 = Image.open(self.template_frames_path_2[item])
        img_rgb_tmp_2 = img_tmp_2.convert('RGB')
        # RGB为彩色图片，L为灰度图片
        if img_rgb_tmp_2.mode != 'RGB':
            raise ValueError("template end image: {} isn't RGB mode.".format(self.template_frames_path_2[item]))


        # tmp_3
        img_tmp_3 = Image.open(self.template_frames_path_3[item])
        img_rgb_tmp_3 = img_tmp_3.convert('RGB')
        # RGB为彩色图片，L为灰度图片
        if img_rgb_tmp_3.mode != 'RGB':
            raise ValueError("template end image: {} isn't RGB mode.".format(self.template_frames_path_3[item]))


        # tmp_4
        img_tmp_4 = Image.open(self.template_frames_path_4[item])
        img_rgb_tmp_4 = img_tmp_4.convert('RGB')
        # RGB为彩色图片，L为灰度图片
        if img_rgb_tmp_4.mode != 'RGB':
            raise ValueError("template end image: {} isn't RGB mode.".format(self.template_frames_path_4[item]))


        img_tmp_5 = Image.open(self.template_frames_path_5[item])
        img_rgb_tmp_5 = img_tmp_5.convert('RGB')
        # RGB为彩色图片，L为灰度图片
        if img_rgb_tmp_5.mode != 'RGB':
            raise ValueError("template end image: {} isn't RGB mode.".format(self.template_frames_path_5[item]))


        matdata = scio.loadmat(self.voxels_path[item])
        # event_features = np.concatenate((matdata['coor'], matdata['features']),
        #                                 axis=1)  # concat coorelate and features (x,y,z, feauture32/16)
        event_features = matdata['event_features']
        # if np.isnan(event_features).any():
        #     event_features = np.zeros(4096, 19)
        #     print('exist nan value in voxel.')
        # # voxel = Image.fromarray(event_features)
        event_features = torch.from_numpy(event_features)
        # z = copy.deepcopy(event_features[:, 0])
        # x, y = event_features[:, 1], event_features[:, 2]
        # event_features[:, 0] = x
        # event_features[:, 1] = y
        # event_features[:, 2] = z
        # event_features = event_features.unsqueeze(0).unsqueeze(0)
        # if event_features.shape[2] < 4096:
        #     pad_len = 4096 - event_features.shape[2]
        # else:
        #     event_features, _ = torch.topk(event_features, k=4096, dim=2)
        #     pad_len = 0
        # event_features = F.pad(event_features.transpose(-1, -2), (0, pad_len), mode='constant', value=0)
        # voxel = event_features.squeeze(0)
        voxel = event_features
        # tmp

        # matdata_tmp = scio.loadmat(self.template_voxels_path[item])

        # event_features_tmp = np.concatenate((matdata_tmp['coor'], matdata_tmp['features']),
        #                                     axis=1)  # concat coorelate and features (x,y,z, feauture32/16)
        
        # event_features_tmp = matdata_tmp['event_features']

        # if np.isnan(event_features_tmp).any():
        #     event_features_tmp = np.zeros(4096, 19)
        #     print('exist nan value in voxel.')
        # voxel_tmp = Image.fromarray(event_features_tmp)
        
        # event_features_tmp = torch.from_numpy(event_features_tmp)
        
        # z = copy.deepcopy(event_features_tmp[:, 0])
        # x, y = event_features_tmp[:, 1], event_features_tmp[:, 2]
        # event_features_tmp[:, 0] = x
        # event_features_tmp[:, 1] = y
        # event_features_tmp[:, 2] = z
        # event_features_tmp = event_features_tmp.unsqueeze(0).unsqueeze(0)
        # if event_features_tmp.shape[2] < 4096:
        #     pad_len = 4096 - event_features_tmp.shape[2]
        # else:
        #     event_features_tmp, _ = torch.topk(event_features_tmp, k=4096, dim=2)
        #     pad_len = 0
        # event_features_tmp = F.pad(event_features_tmp.transpose(-1, -2), (0, pad_len), mode='constant', value=0)
        # voxel_tmp = event_features_tmp.squeeze(0)

        # voxel_tmp = event_features_tmp


        matdata_tmp_1 = scio.loadmat(self.template_voxels_path_1[item])
        event_feature_tmp_1 = matdata_tmp_1['event_features']
        event_feature_tmp_1 = torch.from_numpy(event_feature_tmp_1)
        voxel_tmp_1 = event_feature_tmp_1

        matdata_tmp_2 = scio.loadmat(self.template_voxels_path_2[item])
        event_feature_tmp_2 = matdata_tmp_2['event_features']
        event_feature_tmp_2 = torch.from_numpy(event_feature_tmp_2)
        voxel_tmp_2 = event_feature_tmp_2

        matdata_tmp_3 = scio.loadmat(self.template_voxels_path_3[item])
        event_feature_tmp_3 = matdata_tmp_3['event_features']
        event_feature_tmp_3 = torch.from_numpy(event_feature_tmp_3)
        voxel_tmp_3 = event_feature_tmp_3

        matdata_tmp_4 = scio.loadmat(self.template_voxels_path_4[item])
        event_feature_tmp_4 = matdata_tmp_4['event_features']
        event_feature_tmp_4 = torch.from_numpy(event_feature_tmp_4)
        voxel_tmp_4 = event_feature_tmp_4

        matdata_tmp_5 = scio.loadmat(self.template_voxels_path_5[item])
        event_feature_tmp_5 = matdata_tmp_5['event_features']
        event_feature_tmp_5 = torch.from_numpy(event_feature_tmp_5)
        voxel_tmp_5 = event_feature_tmp_5


        position_label = self.position_label_path[item]
        position_weight_label = self.position_weight_label_path[item]

        # t = transforms.Compose([transforms.RandomResizedCrop(224),
        #                         transforms.RandomHorizontalFlip(),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize([0.485], [0.229])])

        # t = transforms.ToTensor()

        if self.transform is not None:
            img = self.transform(img_rgb)

            img_tmp_1 = self.transform(img_rgb_tmp_1)
            img_tmp_2 = self.transform(img_rgb_tmp_2)
            img_tmp_3 = self.transform(img_rgb_tmp_3)
            img_tmp_4 = self.transform(img_rgb_tmp_4)
            img_tmp_5 = self.transform(img_rgb_tmp_5)

            # voxel = t(voxel)
            # voxel_tmp = t(voxel_tmp)

        return img, voxel, img_tmp_1, voxel_tmp_1, img_tmp_2, voxel_tmp_2, img_tmp_3, voxel_tmp_3, img_tmp_4, voxel_tmp_4, img_tmp_5, voxel_tmp_5, position_label, position_weight_label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        frames, voxels, frames_tmp_1, voxels_tmp_1, frames_tmp_2, voxels_tmp_2, frames_tmp_3, voxels_tmp_3, frames_tmp_4, voxels_tmp_4, frames_tmp_5, voxels_tmp_5, labels, weight_label = tuple(zip(*batch))

        frames = torch.stack(frames, dim=0)
        voxels = torch.stack(voxels, dim=0)

        frames_tmp_1 = torch.stack(frames_tmp_1, dim=0)
        voxels_tmp_1 = torch.stack(voxels_tmp_1, dim=0)

        frames_tmp_2 = torch.stack(frames_tmp_2, dim=0)
        voxels_tmp_2 = torch.stack(voxels_tmp_2, dim=0)

        frames_tmp_3 = torch.stack(frames_tmp_3, dim=0)
        voxels_tmp_3 = torch.stack(voxels_tmp_3, dim=0)

        frames_tmp_4 = torch.stack(frames_tmp_4, dim=0)
        voxels_tmp_4 = torch.stack(voxels_tmp_4, dim=0)

        frames_tmp_5 = torch.stack(frames_tmp_5, dim=0)
        voxels_tmp_5 = torch.stack(voxels_tmp_5, dim=0)

        # frames_tmp_end = torch.stack(frames_tmp_end, dim=0)
        # voxels_tmp_end = torch.stack(voxels_tmp_end, dim=0)
        labels = torch.as_tensor(labels)
        weight_label = torch.as_tensor(weight_label)
        return frames, voxels, frames_tmp_1, voxels_tmp_1, frames_tmp_2, voxels_tmp_2, frames_tmp_3, voxels_tmp_3, frames_tmp_4, voxels_tmp_4, frames_tmp_5, voxels_tmp_5, labels, weight_label
