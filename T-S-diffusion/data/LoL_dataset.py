import os
import torch.utils.data as data
import numpy as np
import torch
import cv2
from torchvision.transforms import ToTensor
import random
import torchvision.transforms as T
import data.util as Util
from scipy import io


class LOLv1_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        if train:
            self.split = 'train'
            #self.root = os.path.join(self.root, 'hr')
            self.root = 'train_data'
        else:
            self.split = 'val'
           # self.root = os.path.join(self.root, 'hr')
            self.root = 'test_data_1'
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        high_list = os.listdir(folder_path)
        # high_list = filter(lambda x: 'png' in x, high_list)

        pairs = []
        for idx, f_name in enumerate(high_list):
            
            if self.split == 'val':
                # pairs.append(
                #     [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  
                #      cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB),
                #     f_name.split('.')[0]])

                pairs.append([io.loadmat(os.path.join(folder_path, f_name))[f_name.split('.')[0]], f_name.split('.')[0]])
                #print(io.loadmat(os.path.join(folder_path, f_name)).keys())
            else:
                # pairs.append(
                #     [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  
                #      cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB),
                #     f_name.split('.')[0]])
                pairs.append([io.loadmat(os.path.join(folder_path, f_name))[f_name.split('.')[0]], f_name.split('.')[0]])
                #print(io.loadmat(os.path.join(folder_path, f_name)).keys())



            #pairs.append([cv2.cvtColor(cv2.imread(os.path.join(folder_path, f_name)), cv2.COLOR_BGR2RGB), f_name.split('.')[0]])
            

        # print(pairs)
        
        return pairs

    def __getitem__(self, item):
        # lr, hr, f_name = self.pairs[item]

        hr, f_name = self.pairs[item]

        # mean = hr.mean(axis=(1, 2))
        # std = hr.std(axis=(1, 2))
        # hr_normalized = (hr  - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]

        X_min = hr.min()
        X_max = hr.max()
        hr_normalized = (hr - X_min) * (1 / (X_max - X_min))


        hr = hr_normalized.transpose(1, 2, 0)
        #print(hr.shape)



        # if self.use_crop and self.split != 'val':
        #     hr, lr = random_crop(hr, lr, self.crop_size)
        # elif self.use_crop and self.split == 'val':
        #     hr = hr[8:392, 12:588, :]
        #     lr = lr[8:392, 12:588, :]

        # elif self.use_crop and self.split == 'val':
        #     h = int(hr.shape[0]/32)
        #     w = int(hr.shape[1]/32)
        #     # print(hr.shape)
        #     # print(h,w)
        #     # print(int((hr.shape[0]-h*32)/2), int(hr.shape[0]-(hr.shape[0]-h*32)/2))
        #     # assert(1==2)
        #     hr = hr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]
        #     lr = lr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]

        # if self.center_crop_hr_size:
        #     hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size)

        # if self.use_flip:
        #     hr, lr = random_flip(hr, lr)

        # if self.use_rot:
        #     hr, lr = random_rotation(hr, lr)
        if self.split == 'val':
            # mean = 0
            # var = 0.0001
            # noise = np.random.normal(mean, var**0.5, hr.shape)
            lr = hr 
            hr = self.to_tensor(hr)
            lr = self.to_tensor(lr)

            [lr, hr] = Util.transform_augment(
                [lr, hr], split=self.split, min_max=(-1, 1))
            
            sigma = 20 / 255.
            noise = np.random.randn(*hr.shape) * sigma

            # lr = hr + noise

        else:
            # mean = 0
            # var = 0.0001
            # noise = np.random.normal(mean, var**0.5, hr.shape)

            lr = hr

            hr = self.to_tensor(hr)
            lr = self.to_tensor(lr)

            [lr, hr] = Util.transform_augment(
                [lr, hr], split=self.split, min_max=(-1, 1))
            
            sigma = 20 / 255.
            noise = np.random.randn(*hr.shape) * sigma

            # lr = hr + noise

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}
        

class LOLv2_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        self.pairs = []
        self.train = train
        for sub_data in ['Synthetic']:  # ['Real_captured, Synthetic']: # :
            if train:
                self.split = 'train'
                root = os.path.join(self.root, sub_data, 'Train')
            else:
                self.split = 'val'
                root = os.path.join(self.root, sub_data, 'Test-10')
            self.pairs.extend(self.load_pairs(root))
        self.to_tensor = ToTensor()
        self.gamma_aug = opt['gamma_aug'] if 'gamma_aug' in opt.keys() else False

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        low_list = os.listdir(os.path.join(folder_path, 'Low' if self.train else 'Low'))
        low_list = sorted(list(filter(lambda x: 'png' in x, low_list)))
        high_list = os.listdir(os.path.join(folder_path, 'Normal' if self.train else 'Normal'))
        high_list = sorted(list(filter(lambda x: 'png' in x, high_list)))
        pairs = []

        for idx in range(len(low_list)):
            f_name_low = low_list[idx]
            f_name_high = high_list[idx]
            pairs.append(
                [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'Low' if self.train else 'Low', f_name_low)),
                                cv2.COLOR_BGR2RGB),  
                    cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'Normal' if self.train else 'Normal', f_name_high)),
                                cv2.COLOR_BGR2RGB), 
                    f_name_high.split('.')[0]])
        return pairs

    def __getitem__(self, item):
        
        lr, hr, f_name = self.pairs[item]

        if self.use_crop and self.split != 'val':
            hr, lr = random_crop(hr, lr, self.crop_size)
        # elif self.use_crop and self.split == 'val':
        #     hr = hr[8:392, 12:588, :]
        #     lr = lr[8:392, 12:588, :]

        if self.center_crop_hr_size:
            hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)


        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        
        [lr, hr] = Util.transform_augment(
                [lr, hr], split=self.split, min_max=(-1, 1))

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}


def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 1).copy()
    seg = seg if random_choice else np.flip(seg, 1).copy()

    return img, seg


def gamma_aug(img, gamma=0):
    max_val = img.max()
    img_after_norm = img / max_val
    img_after_norm = np.power(img_after_norm, gamma)
    return img_after_norm * max_val


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(0, 1)).copy()
    seg = np.rot90(seg, random_choice, axes=(0, 1)).copy()
    
    return img, seg


def random_crop(hr, lr, size_hr):
    size_lr = size_hr

    size_lr_x = lr.shape[0]
    size_lr_y = lr.shape[1]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr, :]

    # HR Patch
    start_x_hr = start_x_lr
    start_y_hr = start_y_lr
    hr_patch = hr[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]

    # HisEq Patch
    his_eq_patch = None
    return hr_patch, lr_patch, 


def center_crop(img, size):
    if img is None:
        return None
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[border:-border, border:-border, :]


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]
