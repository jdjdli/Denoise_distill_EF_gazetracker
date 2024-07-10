import os
import sys
import json
import pickle
import random
import cv2
import shutil

import torch
from tqdm import tqdm
import torch.nn as nn

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
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
    val_frames_path = []  # 存储验证集的所有frames路径
    val_voxels_path = []  # 存储验证集的所有voxels路径
    val_position_label = []  # 存储验证集的所有位置label
    train_template_frames_path = []  # 存储训练集每个user中不同眼的第一帧位置作为基准
    train_template_voxels_path = []  # 存储训练集每个user中不同眼的第一帧位置作为基准
    val_template_frames_path = []  # 存储验证集每个user中不同眼的第一帧位置作为基准
    val_template_voxels_path = []  # 存储验证集每个user中不同眼的第一帧位置作为基准
    # every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 按比例随机采样验证样本
    val = random.sample(user_name, k=int(len(user_name) * val_rate))
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

    # point_select = [
    #     (228, 312), (228, 636), (228, 960), (228, 1284), (228, 1608),
    #
    #     (384, 312), (384, 636), (384, 960), (384, 1284), (384, 1608),
    #
    #     (540, 312), (540, 636), (540, 960), (540, 1284), (540, 1608),
    #
    #     (696, 312), (696, 636), (696, 960), (696, 1284), (696, 1608),
    #
    #     (852, 312), (852, 636), (852, 960), (852, 1284), (852, 1608)
    #
    # ]

    for user in user_name:
        eye_list = os.listdir(os.path.join(root, user))
        for eye_num in eye_list:
            # frames_path = os.path.join(os.path.join(root, user), eye_num, 'frames_select')
            voxels_path = os.path.join(os.path.join(root, user), eye_num, 'voxels_regular1')
            # voxels_path = os.path.join(os.path.join(root, user), eye_num, 'voxels_small')
            # frames = [os.path.join(frames_path, i) for i in
            #           os.listdir(frames_path)]
            voxels = [os.path.join(voxels_path, i) for i in
                      os.listdir(voxels_path)]
            # abs_position = [(int(i.split('_')[1]), int(i.split('_')[2])) for i in os.listdir(frames_path)]

            # for i in os.listdir(frames_path):
            #     img_idx = i.split('_')[0]
            #     img = cv2.imread(os.path.join(frames_path, i))
            #     if not os.path.exists(os.path.join(os.path.join(root, user), eye_num, 'frames_ord')):
            #         os.mkdir(os.path.join(os.path.join(root, user), eye_num, 'frames_ord'))
            #     cv2.imwrite(os.path.join(os.path.join(root, user), eye_num, 'frames_ord', '%04d' % int(img_idx) + '.png'), img)

            # frames_ord_path = os.path.join(os.path.join(root, user), eye_num, 'frames_small_ord')
            frames_ord_path = os.path.join(os.path.join(root, user), eye_num, 'frames_ord1')
            frames = [os.path.join(frames_ord_path, i) for i in
                      os.listdir(frames_ord_path)]
            frames.sort()
            frames_ord = [i for i in os.listdir(frames_ord_path)]
            frames_ord.sort()
            # abs_position = [(int(i.split('_')[1]), int(i.split('_')[2].split('.')[0])) for i in os.listdir(frames_ord_path)]
            # for i in frames_ord:
            #     print(i)
            abs_position = [(int(i.split('_')[1]), int(i.split('_')[2].split('.')[0])) for i in frames_ord]
            
            voxels.sort()
            sample_pair_num.append(len(frames))

            for i in range(len(frames)):
                if user not in val:  # 如果该路径不在采样的验证集样本中则存入训练集
                    train_frames_path.append(frames[i])
                    train_voxels_path.append(voxels[i])
                    position_label = point_select.index(abs_position[i])
                    train_position_label.append(position_label)
                    train_template_frames_path.append(frames[1])
                    train_template_voxels_path.append(voxels[1])
                else:  # 否则存入训练集
                    val_frames_path.append(frames[i])
                    val_voxels_path.append(voxels[i])
                    position_label = point_select.index(abs_position[i])
                    val_position_label.append(position_label)
                    val_template_frames_path.append(frames[1])
                    val_template_voxels_path.append(voxels[1])

    print("{} pairs of data were found in the dataset.".format(sum(sample_pair_num)))
    print("{} pairs of data for training.".format(len(train_frames_path)))
    print("{} pairs of data for validation.".format(len(val_frames_path)))
    assert len(train_frames_path) > 0, "number of training data must greater than 0."
    assert len(val_frames_path) > 0, "number of validation data must greater than 0."

    return train_frames_path, train_voxels_path, train_template_frames_path, train_template_voxels_path, train_position_label, val_frames_path, val_voxels_path, val_template_frames_path, val_template_voxels_path, val_position_label


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


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        frames, voxels, frames_tmp, voxels_tmp, labels = data
        sample_num += frames.shape[0]
        # pred_tmp = model(frames_tmp.to(device), voxels_tmp.to(device))
        pred = model(frames.to(device), voxels.to(device), frames_tmp.to(device), voxels_tmp.to(device))

        # relation = nn.Parameter(torch.zeros(1, 768, 197))
        # nn.init.trunc_normal_(relation, std=0.02)
        # head = nn.Linear(768, 45)
        # pre_logits = nn.Identity()
        # m = pre_logits((pred @ relation) @ pred_tmp)
        # pred = head(m[:, 0])

        pred_classes = torch.max(pred[0], dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()

        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # print(f"Batch {step + 1} gradients:")
        # for j in range(frames.size(0)):
        #     print(f"Sample {j + 1} gradients:")
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             print(f"{name}: {param.grad}")

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        frames, voxels, frames_tmp, voxels_tmp, labels = data
        sample_num += frames.shape[0]
        # pred_tmp = model(frames_tmp.to(device), voxels_tmp.to(device))
        pred = model(frames.to(device), voxels.to(device),frames_tmp.to(device), voxels_tmp.to(device))

        # relation = nn.Parameter(torch.zeros(1, 768, 197))
        # nn.init.trunc_normal_(relation, std=0.02)
        # head = nn.Linear(768, 45)
        # pre_logits = nn.Identity()
        # m = pre_logits((pred @ relation) @ pred_tmp)
        # pred = head(m[:, 0])

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
