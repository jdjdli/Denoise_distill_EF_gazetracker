import os
import sys
import json
import pickle
import random
import cv2
import shutil
from OLE import *
import numpy
import csv
import torch
from tqdm import tqdm
import torch.nn as nn
# from .multi_train_utils.distributed_utils import reduce_value
from multi_train_utils.distributed_utils import reduce_value, is_main_process
# sys.path.append("..")

import matplotlib.pyplot as plt
import feature_denoise
from scipy import io
import scipy

import os
from os.path import basename
import math
import argparse
import random
import logging
import cv2
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import options.options as option
from utils import util
from data import create_dataloader
from data.LoL_dataset import LOLv1_Dataset, LOLv2_Dataset
import torchvision.transforms as T
import lpips
import model as Model
import core.logger as Logger
import core.metrics as Metrics


def read_split_data(root: str, val_rate: float = 0.1):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个user
    user_name = [user for user in os.listdir(root) if os.path.isdir(os.path.join(root, user))]
    # 遍历图片，一个凝视位置对应一个class

    # 排序，保证各平台顺序一致
    user_name.sort()
    # # 生成类别名称以及对应的数字索引
    # class_indices = dict((k, v) for v, k in enumerate(flower_class))
    # json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)
    sample_pair_num = []
    train_frames_path = []  # 存储训练集的所有frames路径
    train_voxels_path = []  # 存储训练集的所有voxels路径
    train_position_label = []  # 存储训练集的所有位置label
    train_position_weight_label = []


    val_frames_path = []  # 存储验证集的所有frames路径
    val_voxels_path = []  # 存储验证集的所有voxels路径
    val_position_label = []  # 存储验证集的所有位置label
    val_position_weight_label = []


    train_template_frames_path_1 = []  # 存储训练集每个user中不同眼的第一帧位置作为基准
    train_template_voxels_path_1 = []  # 存储训练集每个user中不同眼的第一帧位置作为基准
    val_template_frames_path_1 = []  # 存储验证集每个user中不同眼的第一帧位置作为基准
    val_template_voxels_path_1 = []  # 存储验证集每个user中不同眼的第一帧位置作为基准

    train_template_frames_path_2 = []  # 存储训练集每个user中不同眼的第一帧位置作为基准
    train_template_voxels_path_2 = []  # 存储训练集每个user中不同眼的第一帧位置作为基准
    val_template_frames_path_2 = []  # 存储验证集每个user中不同眼的第一帧位置作为基准
    val_template_voxels_path_2 = []  # 存储验证集每个user中不同眼的第一帧位置作为基准

    train_template_frames_path_3 = []  # 存储训练集每个user中不同眼的第一帧位置作为基准
    train_template_voxels_path_3 = []  # 存储训练集每个user中不同眼的第一帧位置作为基准
    val_template_frames_path_3 = []  # 存储验证集每个user中不同眼的第一帧位置作为基准
    val_template_voxels_path_3 = []  # 存储验证集每个user中不同眼的第一帧位置作为基准

    train_template_frames_path_4 = []  # 存储训练集每个user中不同眼的最后一帧位置作为基准
    train_template_voxels_path_4 = []  # 存储训练集每个user中不同眼的最后一帧位置作为基准
    val_template_frames_path_4 = []  # 存储验证集每个user中不同眼的最后一帧位置作为基准
    val_template_voxels_path_4 = []  # 存储验证集每个user中不同眼的最后一帧位置作为基准

    train_template_frames_path_5 = []  # 存储训练集每个user中不同眼的最后一帧位置作为基准
    train_template_voxels_path_5 = []  # 存储训练集每个user中不同眼的最后一帧位置作为基准
    val_template_frames_path_5 = []  # 存储验证集每个user中不同眼的最后一帧位置作为基准
    val_template_voxels_path_5 = []  # 存储验证集每个user中不同眼的最后一帧位置作为基准
    # every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # # 按比例随机采样验证样本
    # val = random.sample(user_name, k=int(len(user_name) * val_rate))
    # 遍历每个文件夹下的文件
    point_select = [(150, 150), (150, 312), (150, 474), (150, 636), (150, 798), (150, 960), (150, 1122), (150, 1284), (150, 1446), (150, 1608), (150, 1770),
                    (228, 150), (228, 312), (228, 474), (228, 636), (228, 798), (228, 960), (228, 1122), (228, 1284), (228, 1446), (228, 1608), (228, 1770),
                    (306, 150), (306, 312), (306, 474), (306, 636), (306, 798), (306, 960), (306, 1122), (306, 1284), (306, 1446), (306, 1608), (306, 1770),
                    (384, 150), (384, 312), (384, 474), (384, 636), (384, 798), (384, 960), (384, 1122), (384, 1284), (384, 1446), (384, 1608), (384, 1770),
                    (462, 150), (462, 312), (462, 474), (462, 636), (462, 798), (462, 960), (462, 1122), (462, 1284), (462, 1446), (462, 1608), (462, 1770),
                    (540, 150), (540, 312), (540, 474), (540, 636), (540, 798), (540, 960), (540, 1122), (540, 1284), (540, 1446), (540, 1608), (540, 1770),
                    (618, 150), (618, 312), (618, 474), (618, 636), (618, 798), (618, 960), (618, 1122), (618, 1284), (618, 1446), (618, 1608), (618, 1770),
                    (696, 150), (696, 312), (696, 474), (696, 636), (696, 798), (696, 960), (696, 1122), (696, 1284), (696, 1446), (696, 1608), (696, 1770),
                    (774, 150), (774, 312), (774, 474), (774, 636), (774, 798), (774, 960), (774, 1122), (774, 1284), (774, 1446), (774, 1608), (774, 1770),
                    (852, 150), (852, 312), (852, 474), (852, 636), (852, 798), (852, 960), (852, 1122), (852, 1284), (852, 1446), (852, 1608), (852, 1770),
                    (930, 150), (930, 312), (930, 474), (930, 636), (930, 798), (930, 960), (930, 1122), (930, 1284), (930, 1446), (930, 1608), (930, 1770)
                    ]

    for user in user_name:
        eye_list = os.listdir(os.path.join(root, user))
        for eye_num in eye_list:
            if eye_num == '0':
                voxels_path = os.path.join(os.path.join(root, user), eye_num, 'voxels_regular1')
            # voxels_path = os.path.join(os.path.join(root, user), eye_num, 'voxels_small')
            # frames = [os.path.join(frames_path, i) for i in
            #           os.listdir(frames_path)]
                voxels = [os.path.join(voxels_path, i) for i in
                      os.listdir(voxels_path)]
            # frames_path = os.path.join(os.path.join(root, user), eye_num, 'frames_select')
            # voxels_path = os.path.join(os.path.join(root, user), eye_num, 'voxels_regular1')
            # frames = [os.path.join(frames_path, i) for i in
            #           os.listdir(frames_path)]
            # voxels = [os.path.join(voxels_path, i) for i in
            #           os.listdir(voxels_path)]
            # abs_position = [(int(i.split('_')[1]), int(i.split('_')[2])) for i in os.listdir(frames_path)]

            # for i in os.listdir(frames_path):
            #     img_idx = i.split('_')[0]
            #     img = cv2.imread(os.path.join(frames_path, i))
            #     if not os.path.exists(os.path.join(os.path.join(root, user), eye_num, 'frames_ord')):
            #         os.mkdir(os.path.join(os.path.join(root, user), eye_num, 'frames_ord'))
            #     cv2.imwrite(os.path.join(os.path.join(root, user), eye_num, 'frames_ord', '%04d' % int(img_idx) + '.png'), img)

            # frames_ord_path = os.path.join(os.path.join(root, user), eye_num, 'frames_ord')
            # frames = [os.path.join(frames_ord_path, i) for i in
            #           os.listdir(frames_ord_path)]
            

                frames_ord_path = os.path.join(os.path.join(root, user), eye_num, 'frames_ord1')
                frames = [os.path.join(frames_ord_path, i) for i in
                      os.listdir(frames_ord_path)]
                frames.sort()
                voxels.sort()
                frames_ord = [i for i in os.listdir(frames_ord_path)]
                frames_ord.sort()

            # abs_position = [(int(i.split('_')[1]), int(i.split('_')[2].split('.')[0])) for i in os.listdir(frames_ord_path)]
                abs_position = [(int(i.split('_')[1]), int(i.split('_')[2].split('.')[0])) for i in frames_ord]



                # position_select = [i for i, x in enumerate(abs_position) if x in [(150, 150), (150, 312), (150, 474), (150, 636), (150, 798), (150, 1122), (150, 1284), (150, 1446), (150, 1608), (150, 1770),
                #                                                                 (228, 150), (228, 312), (228, 474), (228, 636), (228, 798), (228, 1122), (228, 1284), (228, 1446), (228, 1608), (228, 1770),
                #                                                                 (306, 150), (306, 312), (306, 474), (306, 636), (306, 798), (306, 1122), (306, 1284), (306, 1446), (306, 1608), (306, 1770),
                #                                                                 (384, 150), (384, 312), (384, 474), (384, 636), (384, 798), (384, 1122), (384, 1284), (384, 1446), (384, 1608), (384, 1770),
                #                                                                 (462, 150), (462, 312), (462, 474), (462, 636), (462, 798), (462, 1122), (462, 1284), (462, 1446), (462, 1608), (462, 1770),
                                                                                
                #                                                                 (618, 150), (618, 312), (618, 474), (618, 636), (618, 798), (618, 1122), (618, 1284), (618, 1446), (618, 1608), (618, 1770),
                #                                                                 (696, 150), (696, 312), (696, 474), (696, 636), (696, 798), (696, 1122), (696, 1284), (696, 1446), (696, 1608), (696, 1770),
                #                                                                 (774, 150), (774, 312), (774, 474), (774, 636), (774, 798), (774, 1122), (774, 1284), (774, 1446), (774, 1608), (774, 1770),
                #                                                                 (852, 150), (852, 312), (852, 474), (852, 636), (852, 798), (852, 1122), (852, 1284), (852, 1446), (852, 1608), (852, 1770),
                #                                                                 (930, 150), (930, 312), (930, 474), (930, 636), (930, 798), (930, 1122), (930, 1284), (930, 1446), (930, 1608), (930, 1770)]]

                # frames_select = [frames[i] for i in position_select]

                # voxels_select = [voxels[i] for i in position_select]

                # abs_position_select = [abs_position[i] for i in position_select]


        

                position_1 = random.choice([i for i, x in enumerate(abs_position) if x == (150, 150)])
                position_2 = random.choice([i for i, x in enumerate(abs_position) if x == (150, 1770)])
                position_3 = random.choice([i for i, x in enumerate(abs_position) if x == (930, 150)])
                position_4 = random.choice([i for i, x in enumerate(abs_position) if x == (930, 1770)])
                position_5 = random.choice([i for i, x in enumerate(abs_position) if x == (540, 960)])

                
                
                # position_1 = abs_position.index((150, 150))#左上
                # position_2 = abs_position.index((150, 1770))#右上
                # position_3 = abs_position.index((930, 150))#左下
                # position_4 = abs_position.index((930, 1770))#右下


                # indices = [i for i, x in enumerate(my_list) if x == 3]
                # print(indices)  # 输出[2, 4, 7, 10]

                vector_1 = numpy.array([150, 150])
                vector_2 = numpy.array([150, 1770])
                vector_3 = numpy.array([930, 150])
                vector_4 = numpy.array([930, 1770])

            
                
                # sample_pair_num.append(len(frames_select))
                sample_pair_num.append(len(frames))

            # 按比例随机采样验证样本
                # val = random.sample(frames_select, k=int(len(frames_select) * val_rate))
                val = random.sample(frames, k=int(len(frames) * val_rate))
            # print(len(val))

                # for i in range(len(frames_select)):
                for i in range(len(frames)):
                    # if frames_select[i] not in val:  如果该路径不在采样的验证集样本中则存入训练集
                    if frames[i] not in val:
                        # train_frames_path.append(frames_select[i])
                        # train_voxels_path.append(voxels_select[i])
                        # position_label = point_select.index(abs_position_select[i])

                        train_frames_path.append(frames[i])
                        train_voxels_path.append(voxels[i])
                        position_label = point_select.index(abs_position[i])

                        # l = [0] * 121
                        # l[position_label] = 1

                        # train_position_label.append(l)

                        train_position_label.append(position_label)


                        # vector = numpy.array([abs_position_select[i][0], abs_position_select[i][1]])
                        vector = numpy.array([abs_position[i][0], abs_position[i][1]])

                        p1 = numpy.sqrt(sum(numpy.power((vector - vector_1), 2)))
                        p2 = numpy.sqrt(sum(numpy.power((vector - vector_2), 2)))
                        p3 = numpy.sqrt(sum(numpy.power((vector - vector_3), 2)))
                        p4 = numpy.sqrt(sum(numpy.power((vector - vector_4), 2)))

                        p = [p1, p2, p3, p4]
                    # p = [p1, p4]
                        q = [0, 0, 0, 0, 0]

                        if abs_position[i][0] == 540 or abs_position[i][1] == 960:
                            q = [0, 0, 0, 0, 1]
                        else:
                            q[p.index(min(p))] = 1


                        train_position_weight_label.append(q)

                        train_template_frames_path_1.append(frames[position_1])  #左上
                        train_template_voxels_path_1.append(voxels[position_1]) 

                        train_template_frames_path_2.append(frames[position_2])  #右上
                        train_template_voxels_path_2.append(voxels[position_2])

                        train_template_frames_path_3.append(frames[position_3])  #左下
                        train_template_voxels_path_3.append(voxels[position_3])

                        train_template_frames_path_4.append(frames[position_4])  #右下
                        train_template_voxels_path_4.append(voxels[position_4])  

                        train_template_frames_path_5.append(frames[position_5])  #右下
                        train_template_voxels_path_5.append(voxels[position_5]) 

                    # train_template_frames_path.append(frames[1])
                    # train_template_voxels_path.append(voxels[1])
                    # train_template_end_frames_path.append(frames[-1])
                    # train_template_end_voxels_path.append(voxels[-1])
                    else:  # 否则存入训练集
                        # val_frames_path.append(frames_select[i])
                        # val_voxels_path.append(voxels_select[i])
                        # position_label = point_select.index(abs_position_select[i])

                        val_frames_path.append(frames[i])
                        val_voxels_path.append(voxels[i])
                        position_label = point_select.index(abs_position[i])

                        # l = [0] * 121
                        # l[position_label] = 1

                        # val_position_label.append(l)
                        val_position_label.append(position_label)

                        # vector = numpy.array([abs_position_select[i][0], abs_position_select[i][1]])
                        vector = numpy.array([abs_position[i][0], abs_position[i][1]])

                        p1 = numpy.sqrt(sum(numpy.power((vector - vector_1), 2)))
                        p2 = numpy.sqrt(sum(numpy.power((vector - vector_2), 2)))
                        p3 = numpy.sqrt(sum(numpy.power((vector - vector_3), 2)))
                        p4 = numpy.sqrt(sum(numpy.power((vector - vector_4), 2)))

                        p = [p1, p2, p3, p4]
                    # p = [p1, p4]

                        q = [0, 0, 0, 0, 0]

                        if abs_position[i][0] == 540 or abs_position[i][1] == 960:
                            q = [0, 0, 0, 0, 1]
                        else:
                            q[p.index(min(p))] = 1
                        


                        val_position_weight_label.append(q)


                        val_template_frames_path_1.append(frames[position_1])  #左上
                        val_template_voxels_path_1.append(voxels[position_1]) 

                        val_template_frames_path_2.append(frames[position_2])  #右上
                        val_template_voxels_path_2.append(voxels[position_2])

                        val_template_frames_path_3.append(frames[position_3])  #左下
                        val_template_voxels_path_3.append(voxels[position_3])

                        val_template_frames_path_4.append(frames[position_4])  #右下
                        val_template_voxels_path_4.append(voxels[position_4])

                        val_template_frames_path_5.append(frames[position_5])  #右下
                        val_template_voxels_path_5.append(voxels[position_5])


                    # val_template_frames_path.append(frames[1])
                    # val_template_voxels_path.append(voxels[1])
                    # val_template_end_frames_path.append(frames[-1])
                    # val_template_end_voxels_path.append(voxels[-1])

    print("{} pairs of data were found in the dataset.".format(sum(sample_pair_num)))
    print("{} pairs of data for training.".format(len(train_frames_path)))
    print("{} pairs of data for validation.".format(len(val_frames_path)))
    assert len(train_frames_path) > 0, "number of training data must greater than 0."
    assert len(val_frames_path) > 0, "number of validation data must greater than 0."

    return train_frames_path, train_voxels_path, train_template_frames_path_1, train_template_voxels_path_1, train_template_frames_path_2, train_template_voxels_path_2, train_template_frames_path_3, train_template_voxels_path_3, train_template_frames_path_4, train_template_voxels_path_4, train_template_frames_path_5, train_template_voxels_path_5, train_position_label, train_position_weight_label, val_frames_path, val_voxels_path, val_template_frames_path_1, val_template_voxels_path_1, val_template_frames_path_2, val_template_voxels_path_2, val_template_frames_path_3, val_template_voxels_path_3, val_template_frames_path_4, val_template_voxels_path_4, val_template_frames_path_5, val_template_voxels_path_5, val_position_label, val_position_weight_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list
    


# 定义损失函数
def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
    soft_loss = nn.KLDivLoss()(nn.functional.log_softmax(student_logits/T, dim=1),
                                nn.functional.softmax(teacher_logits/T, dim=1))
    hard_loss = nn.CrossEntropyLoss()(student_logits, true_labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super(Invertible1x1Conv, self).__init__()
        # 随机初始化权重
        W = torch.randn(num_channels, num_channels)
        # LU分解可以确保初始化的矩阵是可逆的
        W, _ = torch.linalg.qr(W)
        W = W.unsqueeze(2).unsqueeze(3)
        self.W = nn.Parameter(W)
        self.W_inv = None
        self._log_det_W = None

    def _compute_W_inverse(self):
        if self.W_inv is None:
            W_mat = self.W.squeeze().view(self.W.size(0), -1)
            self.W_inv = torch.inverse(W_mat).view(self.W.size())
            self._log_det_W = torch.slogdet(W_mat)[1]

    def forward(self, x):
        self._compute_W_inverse()
        B, C, H, W = x.size()
        out = nn.functional.conv2d(x, self.W)
        log_det_jacobian = H * W * self._log_det_W
        return out, log_det_jacobian

    def inverse(self, out):
        self._compute_W_inverse()
        return nn.functional.conv2d(out, self.W_inv)



def train_one_epoch(model, optimizer, data_loader, device, epoch, teacher1, teacher2, teacher3, teacher4, teacher5, diffusion):
    model.train()
    teacher1.eval()
    teacher2.eval()
    teacher3.eval()
    teacher4.eval()
    teacher5.eval()
    
    use_OLE = True
    if use_OLE:
        criterion = [nn.CrossEntropyLoss()] + [OLELoss(lambda_=0.25).apply]
    else:
        criterion = [nn.CrossEntropyLoss()]


    loss_function_mlp = nn.CrossEntropyLoss()

    # loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    mlp_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

    dic = {}
    for cnt in range(121):
        dic[cnt] = 0

    # weight_dict = {}

    optimizer.zero_grad()

    accumulate_steps = 16  # 定义要累积的步骤数量

    sample_num = 0
    if is_main_process:
        data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        frames, voxels, frames_tmp_1, voxels_tmp_1, frames_tmp_2, voxels_tmp_2, frames_tmp_3, voxels_tmp_3, frames_tmp_4, voxels_tmp_4, frames_tmp_5, voxels_tmp_5, labels, weight_label = data
        frames, voxels, frames_tmp_1, voxels_tmp_1, frames_tmp_2, voxels_tmp_2, frames_tmp_3, voxels_tmp_3, frames_tmp_4, voxels_tmp_4, frames_tmp_5, voxels_tmp_5, labels, weight_label = torch.autograd.Variable(frames), torch.autograd.Variable(voxels), torch.autograd.Variable(frames_tmp_1), torch.autograd.Variable(voxels_tmp_1), torch.autograd.Variable(frames_tmp_2), torch.autograd.Variable(voxels_tmp_2), torch.autograd.Variable(frames_tmp_3), torch.autograd.Variable(voxels_tmp_3),torch.autograd.Variable(frames_tmp_4), torch.autograd.Variable(voxels_tmp_4), torch.autograd.Variable(frames_tmp_5), torch.autograd.Variable(voxels_tmp_5), torch.autograd.Variable(labels), torch.autograd.Variable(weight_label)
        sample_num += frames.shape[0]
        # pred_tmp = model(frames_tmp.to(device), voxels_tmp.to(device))
        pred = model(frames.to(device), voxels.to(device), frames_tmp_1.to(device), voxels_tmp_1.to(device), frames_tmp_2.to(device), voxels_tmp_2.to(device), frames_tmp_3.to(device), voxels_tmp_3.to(device), frames_tmp_4.to(device), voxels_tmp_4.to(device), frames_tmp_5.to(device), voxels_tmp_5.to(device))

        teacher1_logits = teacher1(frames.to(device), voxels.to(device), frames_tmp_1.to(device), voxels_tmp_1.to(device))
        teacher2_logits = teacher2(frames.to(device), voxels.to(device), frames_tmp_2.to(device), voxels_tmp_2.to(device))
        teacher3_logits = teacher3(frames.to(device), voxels.to(device), frames_tmp_3.to(device), voxels_tmp_3.to(device))
        teacher4_logits = teacher4(frames.to(device), voxels.to(device), frames_tmp_4.to(device), voxels_tmp_4.to(device))
        teacher5_logits = teacher5(frames.to(device), voxels.to(device), frames_tmp_5.to(device), voxels_tmp_5.to(device))

        avg_teacher_logits = (teacher1_logits[2] + teacher2_logits[2] + teacher3_logits[2] + teacher4_logits[2] + teacher5_logits[2]) / 5
        
        avg_teacher_logits_feature = (teacher1_logits[3] + teacher2_logits[3] + teacher3_logits[3] + teacher4_logits[3] + teacher5_logits[3]) / 5

        batch_numpy = avg_teacher_logits_feature.reshape([avg_teacher_logits_feature.size(dim=0), 3, 16, 16, 14, 14]).transpose(4, 3)
        
        batch_numpy= batch_numpy.reshape([pred[4].size(dim=0), 3, 224, 224])


        batch_numpy_normalized = (batch_numpy - batch_numpy.min()) * (1 / (batch_numpy.max() - batch_numpy.min()))


        diffusion.feed_data(batch_numpy_normalized)
        diffusion.test(continous=False)

        visuals = diffusion.get_current_visuals()

        generate_feature = visuals['HQ'].to(device) * (batch_numpy.max() - batch_numpy.min()) + batch_numpy.min()

        # print(type(generate_feature))

        generate_feature = generate_feature.reshape([avg_teacher_logits_feature.size(dim=0), 3, 16, 14, 16, 14]).transpose(4, 3)
        generate_feature = generate_feature.reshape([avg_teacher_logits_feature.size(dim=0), 768, 14, 14]).to(device)

        

        
        # with torch.no_grad():
        #     batch_numpy = avg_teacher_logits_feature.reshape([avg_teacher_logits_feature.size(dim=0), 3, 16, 16, 14, 14]).transpose(4, 3)
        
        #     batch_numpy= batch_numpy.reshape([pred[4].size(dim=0), 3, 224, 224]).cpu().detach().numpy()
        #     new_feature = []
        #     for i in range(pred[4].size(dim=0)):
        #         scipy.io.savemat(f'test_data_1/test_step_{i}_data.mat', {f'test_step_{i}_data': batch_numpy[i]})
        #         generate_feature = feature_denoise.feature_denoise().detach()
        #         generate_feature = generate_feature.reshape([3, 16, 14, 16, 14]).transpose(3, 2)
        #         generate_feature = generate_feature.reshape([768, 14, 14]).unsqueeze(0)
        #         new_feature.append(generate_feature)

        # # try:
        # #     my_tensor = torch.tensor(new_feature)
        # # except ValueError as e:
        # #     print(e)  # sizes must be consistent
        # my_tensor = torch.stack(new_feature, dim=0)
        # print(type(avg_teacher_logits_feature))
        # print(type(new_feature))
        # print(type(my_tensor))
        # new_feature_from_constructor = my_tensor.to(device)
        new_feature_from_constructor = generate_feature


        # 使用示例
    #     num_channels = 768  # 输入具有768个通道
    #     invertible_conv = Invertible1x1Conv(num_channels).to(device)
    #     # 正向传播
    #     output_teacher, log_det_jacobian_teacher = invertible_conv(avg_teacher_logits_feature)
    #     # 正向传播
    #     output_student, log_det_jacobian_student = invertible_conv(pred[4])

    #     output_teacher1, log_det_jacobian_teacher1 = invertible_conv(teacher1_logits[3])
    #     output_teacher2, log_det_jacobian_teacher2 = invertible_conv(teacher2_logits[3])
    #     output_teacher3, log_det_jacobian_teacher3 = invertible_conv(teacher3_logits[3])
    #     output_teacher4, log_det_jacobian_teacher4 = invertible_conv(teacher4_logits[3])
    #     output_teacher5, log_det_jacobian_teacher5 = invertible_conv(teacher5_logits[3])

    #     norm_teacher = torch.norm(output_teacher, p=2)
    #     norm_student = torch.norm(output_student, p=2)
    #    # loss_new = torch.abs(norm_student - norm_teacher)
    #     loss_new = torch.norm(output_teacher - output_student, p=2)

        # weight_dict['teacher1_attn:'] = teacher1_logits[2]
        # weight_dict['teacher2_attn:'] = teacher2_logits[2]
        # weight_dict['teacher3_attn:'] = teacher3_logits[2]
        # weight_dict['teacher4_attn:'] = teacher4_logits[2]
        # weight_dict['student_attn:'] = pred[3]

        # filename = 'attn_weight.csv'
 
        # # 打开文件并写入数据
        # with open(filename, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(weight_dict)

        # print('teacher1_attn:', teacher1_logits[2])
        # print('teacher2_attn:', teacher2_logits[2])
        # print('teacher3_attn:', teacher3_logits[2])
        # print('teacher4_attn:', teacher4_logits[2])
        # print('student_attn:', pred[3])
        # relation = nn.Parameter(torch.zeros(1, 768, 197))
        # nn.init.trunc_normal_(relation, std=0.02)
        # head = nn.Linear(768, 45)
        # pre_logits = nn.Identity()
        # m = pre_logits((pred @ relation) @ pred_tmp)
        # pred = head(m[:, 0])


        # criterion is a list composed of crossentropy loss and OLE loss
        losses_list = [-1, -1]
        # output_Var contains scores in the first element and features in the second element
        loss = 0
        # for cix, loss_function in enumerate(criterion):
        #     losses_list[cix] = loss_function(pred[cix], labels.to(device))
        # if len(criterion) == 2:
        #     loss = losses_list[0] + losses_list[1].data[0]
        # else:
        #     loss = losses_list[0]

        # 超参数 temperature 和 alpha 需要根据您的具体问题进行调整。温度（temperature）用于控制软标签的平滑程度，而 alpha 用于调节软标签和硬标签损失的权重。

        # loss = distillation_loss(student_logits, avg_teacher_logits, labels, T=temperature, alpha=alpha)

        loss_mlp = 0
        loss_mlp = loss_function_mlp(pred[2], weight_label.float().to(device))

        alpha = 1
        T = 4
        # distillation = torch.nn.KLDivLoss(reduction=batchmean)

        soft_loss = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(pred[3]/T, dim=1),
                                nn.functional.softmax(avg_teacher_logits/T, dim=1))
        
        # soft_loss = nn.functional.mse_loss(pred[3]/T, avg_teacher_logits)
        
        # soft_loss_1 = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(pred[4]/T, dim=1),
        #                         nn.functional.softmax(avg_teacher_logits_feature/T, dim=1))
        
        # soft_loss_1 = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(pred[4]/T, dim=1),
        #                         nn.functional.softmax(new_feature_from_constructor/T, dim=1))
        soft_loss_1 = nn.functional.mse_loss(pred[4], new_feature_from_constructor)
        # soft_loss = nn.KLDivLoss(reduction='batchmean')((pred[3]/T), (avg_teacher_logits/T))
        hard_loss = nn.CrossEntropyLoss()(pred[0], labels.to(device))
        loss = alpha * soft_loss + (alpha) * hard_loss + 500 * soft_loss_1 

        print('softloss', soft_loss)
        print('softloss_1', soft_loss_1)
        print('hardloss', hard_loss)

        # loss_mlp = 0
        # loss_mlp = loss_function_mlp(pred[2], weight_label.float().to(device))

        # loss = loss + loss_mlp
        



        pred_classes = torch.max(pred[0], dim=1)[1]
        # labels_classes = torch.max(labels.to(device), dim=1)[1]
        # print(labels.size())
        # os.system('pause')
        # accu_num += torch.eq(pred_classes, labels_classes).sum()
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # accu_num = reduce_value(accu_num, average=False)
        # loss = loss_function(pred, labels.to(device))

        loss = loss / accumulate_steps  # 将损失平均分摊到累积步数

        loss.backward()
        loss = reduce_value(loss, average=True)
        accu_loss += loss.detach()
        # accu_loss += losses_list[0].detach()
        # mlp_loss += loss_mlp.detach()
        mlp_loss += loss_mlp

        # for index in range(len(pred_classes)):
        #     if pred_classes[index] == labels_classes[index]:
        #         # print(pred_classes[index].cpu().detach().numpy())
        #         dic[pred_classes[index].cpu().detach().numpy().tolist()] += 1

        # if is_main_process():
        #     data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                        accu_loss.item() / (step + 1),
        #                                                                        accu_num.item() / sample_num)
        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))
        # data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                        accu_loss.item() / (step + 1),
        #                                                                        accu_num.item() / sample_num)
        


        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # print(f"Batch {step + 1} gradients:")
        # for j in range(frames.size(0)):
        #     print(f"Sample {j + 1} gradients:")
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             print(f"{name}: {param.grad}")
        if (step + 1) % accumulate_steps == 0:  # 每积累一定步数后，更新一次权重
            optimizer.step()
            optimizer.zero_grad()

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    accu_num = reduce_value(accu_num, average=False)
    # data_loader.desc = "[train epoch {}] acc: {:.3f}".format(epoch, accu_num.item() / sample_num)
    a1 = sorted(dic.items(),key = lambda x:x[1],reverse = True)
    # print(a1[0], a1[1], a1[2], a1[3], a1[4])
    # print(a1)
    

    # weight_dict = [['teacher1_attn:', teacher1_logits[2].cpu().detach().numpy()],
    #                ['teacher2_attn:', teacher2_logits[2].cpu().detach().numpy()],
    #                ['teacher3_attn:', teacher3_logits[2].cpu().detach().numpy()],
    #                ['teacher4_attn:', teacher4_logits[2].cpu().detach().numpy()],
    #                ['student_attn:', pred[3].cpu().detach().numpy()]]
    # np.set_printoptions(threshold=np.inf)

    # filename = 'attn_weight.csv'
 
    #     # 打开文件并写入数据
    # with open(filename, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(weight_dict)
    # print('csv was saved completely!')
    # np.set_printoptions(threshold=np.inf)
    # print('teacher1_attn:', teacher1_logits[2].cpu().detach().numpy())
    # print('teacher2_attn:', teacher2_logits[2].cpu().detach().numpy())
    # print('teacher3_attn:', teacher3_logits[2].cpu().detach().numpy())
    # print('teacher4_attn:', teacher4_logits[2].cpu().detach().numpy())
    # print('softloss', soft_loss)
    # print('hardloss', hard_loss)
    # print('avg_teacher_attn:', avg_teacher_logits.cpu().detach().numpy())
    # print('student_attn:', pred[3].cpu().detach().numpy())
    

    # return accu_loss.item() / (step + 1), accu_num.item() / sample_num
    return accu_loss.item() / (step + 1), accu_num.item(), mlp_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    # loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    use_OLE = True
    if use_OLE:
        criterion = [nn.CrossEntropyLoss()] + [OLELoss(lambda_=0.25).apply]
    else:
        criterion = [nn.CrossEntropyLoss()]

    loss_function_mlp = nn.CrossEntropyLoss()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    mlp_loss = torch.zeros(1).to(device)
    theta_deg = []
    pred_num = []

    dic = {}
    for cnt in range(121):
        dic[cnt] = 0

    sample_num = 0
    if is_main_process:
        data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        # frames, voxels, frames_tmp, voxels_tmp, frames_tmp_end, voxels_tmp_end, labels = data
        # frames, voxels, frames_tmp, voxels_tmp, frames_tmp_end, voxels_tmp_end, labels = torch.autograd.Variable(frames), torch.autograd.Variable(voxels), torch.autograd.Variable(frames_tmp), torch.autograd.Variable(voxels_tmp), torch.autograd.Variable(frames_tmp_end), torch.autograd.Variable(voxels_tmp_end), torch.autograd.Variable(labels)
        # sample_num += frames.shape[0]
        # # pred_tmp = model(frames_tmp.to(device), voxels_tmp.to(device))
        # pred = model(frames.to(device), voxels.to(device),frames_tmp.to(device), voxels_tmp.to(device), frames_tmp_end.to(device), voxels_tmp_end.to(device))

        frames, voxels, frames_tmp_1, voxels_tmp_1, frames_tmp_2, voxels_tmp_2, frames_tmp_3, voxels_tmp_3, frames_tmp_4, voxels_tmp_4, frames_tmp_5, voxels_tmp_5, labels, weight_label = data
        frames, voxels, frames_tmp_1, voxels_tmp_1, frames_tmp_2, voxels_tmp_2, frames_tmp_3, voxels_tmp_3, frames_tmp_4, voxels_tmp_4, frames_tmp_5, voxels_tmp_5, labels, weight_label = torch.autograd.Variable(frames), torch.autograd.Variable(voxels), torch.autograd.Variable(frames_tmp_1), torch.autograd.Variable(voxels_tmp_1), torch.autograd.Variable(frames_tmp_2), torch.autograd.Variable(voxels_tmp_2), torch.autograd.Variable(frames_tmp_3), torch.autograd.Variable(voxels_tmp_3),torch.autograd.Variable(frames_tmp_4), torch.autograd.Variable(voxels_tmp_4), torch.autograd.Variable(frames_tmp_5), torch.autograd.Variable(voxels_tmp_5),torch.autograd.Variable(labels), torch.autograd.Variable(weight_label)
        pred = model(frames.to(device), voxels.to(device), frames_tmp_1.to(device), voxels_tmp_1.to(device), frames_tmp_2.to(device), voxels_tmp_2.to(device), frames_tmp_3.to(device), voxels_tmp_3.to(device), frames_tmp_4.to(device), voxels_tmp_4.to(device), frames_tmp_5.to(device), voxels_tmp_5.to(device))



        losses_list = [-1, -1]
        # output_Var contains scores in the first element and features in the second element
        loss = 0
        for cix, loss_function in enumerate(criterion):
            losses_list[cix] = loss_function(pred[cix], labels.to(device))
        if len(criterion) == 2:
            loss = losses_list[0] + losses_list[1].data[0]
        else:
            loss = losses_list[0]

        loss_mlp = 0
        # loss_mlp = loss_function_mlp(pred[2], weight_label.float().to(device))

        # loss = loss + loss_mlp


        # relation = nn.Parameter(torch.zeros(1, 768, 197))
        # nn.init.trunc_normal_(relation, std=0.02)
        # head = nn.Linear(768, 45)
        # pre_logits = nn.Identity()
        # m = pre_logits((pred @ relation) @ pred_tmp)
        # pred = head(m[:, 0])

        pred_classes = torch.max(pred[0], dim=1)[1]
        # labels_classes = torch.max(labels.to(device), dim=1)[1]
        # accu_num += torch.eq(pred_classes, labels_classes).sum()
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()



        point_select = [(150, 150), (150, 312), (150, 474), (150, 636), (150, 798), (150, 960), (150, 1122), (150, 1284), (150, 1446), (150, 1608), (150, 1770),
                    (228, 150), (228, 312), (228, 474), (228, 636), (228, 798), (228, 960), (228, 1122), (228, 1284), (228, 1446), (228, 1608), (228, 1770),
                    (306, 150), (306, 312), (306, 474), (306, 636), (306, 798), (306, 960), (306, 1122), (306, 1284), (306, 1446), (306, 1608), (306, 1770),
                    (384, 150), (384, 312), (384, 474), (384, 636), (384, 798), (384, 960), (384, 1122), (384, 1284), (384, 1446), (384, 1608), (384, 1770),
                    (462, 150), (462, 312), (462, 474), (462, 636), (462, 798), (462, 960), (462, 1122), (462, 1284), (462, 1446), (462, 1608), (462, 1770),
                    (540, 150), (540, 312), (540, 474), (540, 636), (540, 798), (540, 960), (540, 1122), (540, 1284), (540, 1446), (540, 1608), (540, 1770),
                    (618, 150), (618, 312), (618, 474), (618, 636), (618, 798), (618, 960), (618, 1122), (618, 1284), (618, 1446), (618, 1608), (618, 1770),
                    (696, 150), (696, 312), (696, 474), (696, 636), (696, 798), (696, 960), (696, 1122), (696, 1284), (696, 1446), (696, 1608), (696, 1770),
                    (774, 150), (774, 312), (774, 474), (774, 636), (774, 798), (774, 960), (774, 1122), (774, 1284), (774, 1446), (774, 1608), (774, 1770),
                    (852, 150), (852, 312), (852, 474), (852, 636), (852, 798), (852, 960), (852, 1122), (852, 1284), (852, 1446), (852, 1608), (852, 1770),
                    (930, 150), (930, 312), (930, 474), (930, 636), (930, 798), (930, 960), (930, 1122), (930, 1284), (930, 1446), (930, 1608), (930, 1770)
                    ]
        

        point_select_trans = [(40, 16.865, -43.6525), (40, 16.865, -34.922), (40, 16.865, -26.1915), (40, 16.865, -17.461), (40, 16.865, -8.7305), (40, 16.865, 0), (40, 16.865, 8.7305), (40, 16.865, 17.461), (40, 16.865, 26.1915), (40, 16.865, 34.922), (40, 16.865, 43.6525),
                            (40, 11.8056, -43.6525), (40, 11.8056, -34.922), (40, 11.8056, -26.1915), (40, 11.8056, -17.461), (40, 11.8056, -8.7305), (40, 11.8056, 0), (40, 11.8056, 8.7305), (40, 11.8056, 17.461), (40, 11.8056, 26.1915), (40, 11.8056, 34.922), (40, 11.8056, 43.6525),
                            (40, 6.7462, -43.6525), (40, 6.7462, -34.922), (40, 6.7462, -26.1915), (40, 6.7462, -17.461), (40, 6.7462, -8.7305), (40, 6.7462, 0), (40, 6.7462, 8.7305), (40, 6.7462, 17.461), (40, 6.7462, 26.1915), (40, 6.7462, 34.922), (40, 6.7462, 43.6525),
                            (40, 1.6868, -43.6525), (40, 1.6868, -34.922), (40, 1.6868, -26.1915), (40, 1.6868, -17.461), (40, 1.6868, -8.7305), (40, 1.6868, 0), (40, 1.6868, 8.7305), (40, 1.6868, 17.461), (40, 1.6868, 26.1915), (40, 1.6868, 34.922), (40, 1.6868, 43.6525),
                            (40, -3.3726, -43.6525), (40, -3.3726, -34.922), (40, -3.3726, -26.1915), (40, -3.3726, -17.461), (40, -3.3726, -8.7305), (40, -3.3726, 0), (40, -3.3726, 8.7305), (40, -3.3726, 17.461), (40, -3.3726, 26.1915), (40, -3.3726, 34.922), (40, -3.3726, 43.6525),
                            (40, -8.432, -43.6525), (40, -8.432, -34.922), (40, -8.432, -26.1915), (40, -8.432, -17.461), (40, -8.432, -8.7305), (40, -8.432, 0), (40, -8.432, 8.7305), (40, -8.432, 17.461), (40, -8.432, 26.1915), (40, -8.432, 34.922), (40, -8.432, 43.6525),
                            (40, -13.4914, -43.6525), (40, -13.4914, -34.922), (40, -13.4914, -26.1915), (40, -13.4914, -17.461), (40, -13.4914, -8.7305), (40, -13.4914, 0), (40, -13.4914, 8.7305), (40, -13.4914, 17.461), (40, -13.4914, 26.1915), (40, -13.4914, 34.922), (40, -13.4914, 43.6525),
                            (40, -18.5508, -43.6525), (40, -18.5508, -34.922), (40, -18.5508, -26.1915), (40, -18.5508, -17.461), (40, -18.5508, -8.7305), (40, -18.5508, 0), (40, -18.5508, 8.7305), (40, -18.5508, 17.461), (40, -18.5508, 26.1915), (40, -18.5508, 34.922), (40, -18.5508, 43.6525),
                            (40, -23.6102, -43.6525), (40, -23.6102, -34.922), (40, -23.6102, -26.1915), (40, -23.6102, -17.461), (40, -23.6102, -8.7305), (40, -23.6102, 0), (40, -23.6102, 8.7305), (40, -23.6102, 17.461), (40, -23.6102, 26.1915), (40, -23.6102, 34.922), (40, -23.6102, 43.6525),
                            (40, -28.6696, -43.6525), (40, -28.6696, -34.922), (40, -28.6696, -26.1915), (40, -28.6696, -17.461), (40, -28.6696, -8.7305), (40, -28.6696, 0), (40, -28.6696, 8.7305), (40, -28.6696, 17.461), (40, -28.6696, 26.1915), (40, -28.6696, 34.922), (40, -28.6696, 43.6525),
                            (40, -33.729, -43.6525), (40, -33.729, -34.922), (40, -33.729, -26.1915), (40, -33.729, -17.461), (40, -33.729, -8.7305), (40, -33.729, 0), (40, -33.729, 8.7305), (40, -33.729, 17.461), (40, -33.729, 26.1915), (40, -33.729, 34.922), (40, -33.729, 43.6525)
                    ]
        
        for i in range(pred_classes.size()[0]):
            vector1 = numpy.array([point_select_trans[pred_classes[i].to('cpu')][0], point_select_trans[pred_classes[i].to('cpu')][1], point_select_trans[pred_classes[i].to('cpu')][2]])
            vector2 = numpy.array([point_select_trans[labels[i]][0], point_select_trans[labels[i]][1], point_select_trans[labels[i]][2]])

            # 计算点积
            dot_product = np.dot(vector1, vector2)

            # 计算向量的模（范数）
            norm_a = np.linalg.norm(vector1)
            norm_b = np.linalg.norm(vector2)

            # 防止因浮点数精度问题导致arccos函数输入超出范围[-1, 1]
            cos_theta = np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0)

            # 计算夹角的余弦值并求得弧度
            theta_rad = np.arccos(cos_theta)

            # 如果你需要将夹角转换为度数
            theta_deg.append(np.degrees(theta_rad))

            pred_num.append(point_select[labels[i].to('cpu')])

        

        # for index in range(len(pred_classes)):
        #     if pred_classes[index] == labels_classes[index]:
        #         # print(pred_classes[index].cpu().detach().numpy())
        #         dic[pred_classes[index].cpu().detach().numpy().tolist()] += 1



        # loss = loss_function(pred, labels.to(device))
        loss = reduce_value(loss, average=True)
        accu_loss += losses_list[0]
        mlp_loss += loss_mlp

        # if is_main_process:
        data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))
    
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    accu_num = reduce_value(accu_num, average=False)
    # data_loader.desc = "[valid epoch {}] acc: {:.3f}".format(epoch, accu_num.item() / sample_num)
    a1 = sorted(dic.items(),key = lambda x:x[1],reverse = True)
    # print(a1[0], a1[1], a1[2], a1[3], a1[4])
    # print(a1)

    # return accu_loss.item() / (step + 1), accu_num.item() / sample_num
    return accu_loss.item() / (step + 1), accu_num.item() , mlp_loss.item() / (step + 1), theta_deg, pred_num
