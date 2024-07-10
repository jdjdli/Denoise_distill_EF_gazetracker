import os
import math
import argparse
import tempfile

import sys
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# import d2l
import numpy as np
from collections import Counter

from my_dataset_classification_rel import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from teacher_model import vit_base_patch16_224_in21k as teacher_model
# from vit_model_frames import vit_base_patch16_224_in21k as create_model
from evaluate_utils_template_multiGPU import read_split_data, train_one_epoch, evaluate
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
# sys.path.append("..")


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.gpu
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""
    # print(rank)

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs"')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #
    # if os.path.exists("./weights") is False:
    #     os.makedirs("./weights")
    #
    # tb_writer = SummaryWriter()

    # train_frames_path, train_voxels_path, val_frames_path, val_voxels_path = read_split_data(args.data_path)
    train_frames_path, train_voxels_path, train_template_frames_path_1, train_template_voxels_path_1, train_template_frames_path_2, train_template_voxels_path_2, train_template_frames_path_3, train_template_voxels_path_3, train_template_frames_path_4, train_template_voxels_path_4, train_position_label, train_position_weight_label, val_frames_path, val_voxels_path, val_template_frames_path_1, val_template_voxels_path_1, val_template_frames_path_2, val_template_voxels_path_2, val_template_frames_path_3, val_template_voxels_path_3, val_template_frames_path_4, val_template_voxels_path_4, val_position_label, val_position_weight_label = read_split_data(
        args.data_path)

    data_transform = {
        # "train": transforms.Compose([transforms.RandomResizedCrop(224),
        #                              transforms.RandomHorizontalFlip(),
        #                              transforms.ToTensor(),
        #                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "train": transforms.Compose([transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(frames_path=train_frames_path,
                              voxels_path=train_voxels_path, template_frames_path_1=train_template_frames_path_1,
                              template_voxels_path_1=train_template_voxels_path_1, template_frames_path_2=train_template_frames_path_2, template_voxels_path_2=train_template_voxels_path_2, template_frames_path_3=train_template_frames_path_3, template_voxels_path_3=train_template_voxels_path_3, template_frames_path_4=train_template_frames_path_4, template_voxels_path_4=train_template_voxels_path_4, position_label_path=train_position_label, position_weight_label_path=train_position_weight_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(frames_path=val_frames_path,
                            voxels_path=val_voxels_path, template_frames_path_1=val_template_frames_path_1,
                            template_voxels_path_1=val_template_voxels_path_1, template_frames_path_2=val_template_frames_path_2, template_voxels_path_2=val_template_voxels_path_2, template_frames_path_3=val_template_frames_path_3, template_voxels_path_3=val_template_voxels_path_3, template_frames_path_4=val_template_frames_path_4, template_voxels_path_4=val_template_voxels_path_4, position_label_path=val_position_label, position_weight_label_path=val_position_weight_label,
                            transform=data_transform["val"])

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes,
                         has_logits=False).to(device)
    
    # teacher1 = teacher_model(num_classes=args.num_classes, has_logits=False).to(device)
    # teacher2 = teacher_model(num_classes=args.num_classes, has_logits=False).to(device)
    # teacher3 = teacher_model(num_classes=args.num_classes, has_logits=False).to(device)
    # teacher4 = teacher_model(num_classes=args.num_classes, has_logits=False).to(device)

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(args.weights, map_location=device)
        # weights_dict1 = torch.load(args.weights1, map_location=device)

        # weights_teacher1 = torch.load(args.left_top_weights, map_location=device)
        # weights_teacher2 = torch.load(args.right_top_weights, map_location=device)
        # weights_teacher3 = torch.load(args.left_btm_weights, map_location=device)
        # weights_teacher4 = torch.load(args.right_btm_weights, map_location=device)

        # new_state_dict1 = {}
        # for k,v in weights_dict.items():
        #     new_state_dict1[k[7:]] = v

        # new_state_dict = {}
        # for k,v in weights_dict1.items():
        #     new_state_dict[k[7:]] = v


        whole_dict = {}
        for k,v in weights_dict.items():
            whole_dict[k[7:]] = v

        # teacher1_dict = {}
        # for k,v in weights_teacher1.items():
        #     teacher1_dict[k[7:]] = v

        # teacher2_dict = {}
        # for k,v in weights_teacher2.items():
        #     teacher2_dict[k[7:]] = v


        # teacher3_dict = {}
        # for k,v in weights_teacher3.items():
        #     teacher3_dict[k[7:]] = v


        # teacher4_dict = {}
        # for k,v in weights_teacher4.items():
        #     teacher4_dict[k[7:]] = v

        # # new_dict = { 'pos_embed_1' if key == 'pos_embed' else key : value for key, value in new_state_dict.items()}
        # new_dict = { 'patch_embed_1.proj.weight' if key == 'patch_embed.proj.weight' else key : value for key, value in new_state_dict.items()}
        # new_dict = { 'patch_embed_1.proj.bias' if key == 'patch_embed.proj.bias' else key : value for key, value in new_dict.items()}
        # # new_dict = { 'patch_embed_tmp_1.proj.weight' if key == 'patch_embed_tmp.proj.weight' else key : value for key, value in new_dict.items()}
        # # new_dict = { 'patch_embed_tmp_1.proj.bias' if key == 'patch_embed_tmp.proj.bias' else key : value for key, value in new_dict.items()}
        # new_dict = { 'patch_embed_event_1.pos_embedding.weight' if key == 'patch_embed_event.pos_embedding.weight' else key : value for key, value in new_dict.items()}
        # new_dict = { 'patch_embed_event_1.pos_embedding.bias' if key == 'patch_embed_event.pos_embedding.bias' else key : value for key, value in new_dict.items()}
        # new_dict = { 'patch_embed_event_1.proj.weight' if key == 'patch_embed_event.proj.weight' else key : value for key, value in new_dict.items()}
        # new_state_dict = { 'patch_embed_event_1.proj.bias' if key == 'patch_embed_event.proj.bias' else key : value for key, value in new_dict.items()}




        # new_state_dict1.update(new_state_dict)


        # print(weights_dict)
        # os.system('pause')

        # 删除不需要的权重
        # del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        #     else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']

        # del_keys = ['head.weight', 'head.bias'] if model.has_logits else ['pre_logits.fc.weight', 'pre_logits.fc.bias',
        #                                                                   'head.weight',
        #                                                                   'head.bias',
        #                                                                   'blocks.1.norm1.bias', 'blocks.1.norm1.weight', 'blocks.1.norm2.bias', 'blocks.1.norm2.weight', 'blocks.1.mlp.fc1.bias', 'blocks.1.mlp.fc1.weight',
        #                                                                   'blocks.1.mlp.fc2.bias', 'blocks.1.mlp.fc2.weight', 'blocks.1.attn.proj.bias', 'blocks.1.attn.proj.weight', 'blocks.1.attn.qkv.bias', 'blocks.1.attn.qkv.weight',
        #                                                                   'blocks.10.norm1.bias', 'blocks.10.norm1.weight', 'blocks.10.norm2.bias', 'blocks.10.norm2.weight', 'blocks.10.mlp.fc1.bias', 'blocks.10.mlp.fc1.weight',
        #                                                                   'blocks.10.mlp.fc2.bias', 'blocks.10.mlp.fc2.weight', 'blocks.10.attn.proj.bias', 'blocks.10.attn.proj.weight', 'blocks.10.attn.qkv.bias', 'blocks.10.attn.qkv.weight',
        #                                                                   'blocks.11.norm1.bias', 'blocks.11.norm1.weight', 'blocks.11.norm2.bias', 'blocks.11.norm2.weight', 'blocks.11.mlp.fc1.bias', 'blocks.11.mlp.fc1.weight', 
        #                                                                   'blocks.11.mlp.fc2.bias', 'blocks.11.mlp.fc2.weight', 'blocks.11.attn.proj.bias', 'blocks.11.attn.proj.weight', 'blocks.11.attn.qkv.bias', 'blocks.11.attn.qkv.weight', 
        #                                                                   'blocks.2.norm1.bias', 'blocks.2.norm1.weight', 'blocks.2.norm2.bias', 'blocks.2.norm2.weight', 'blocks.2.mlp.fc1.bias', 'blocks.2.mlp.fc1.weight', 
        #                                                                   'blocks.2.mlp.fc2.bias', 'blocks.2.mlp.fc2.weight', 'blocks.2.attn.proj.bias', 'blocks.2.attn.proj.weight', 'blocks.2.attn.qkv.bias', 'blocks.2.attn.qkv.weight', 
        #                                                                   'blocks.3.norm1.bias', 'blocks.3.norm1.weight', 'blocks.3.norm2.bias', 'blocks.3.norm2.weight', 'blocks.3.mlp.fc1.bias', 'blocks.3.mlp.fc1.weight', 
        #                                                                   'blocks.3.mlp.fc2.bias', 'blocks.3.mlp.fc2.weight', 'blocks.3.attn.proj.bias', 'blocks.3.attn.proj.weight', 'blocks.3.attn.qkv.bias', 'blocks.3.attn.qkv.weight', 
        #                                                                   'blocks.4.norm1.bias', 'blocks.4.norm1.weight', 'blocks.4.norm2.bias', 'blocks.4.norm2.weight', 'blocks.4.mlp.fc1.bias', 'blocks.4.mlp.fc1.weight', 
        #                                                                   'blocks.4.mlp.fc2.bias', 'blocks.4.mlp.fc2.weight', 'blocks.4.attn.proj.bias', 'blocks.4.attn.proj.weight', 'blocks.4.attn.qkv.bias', 'blocks.4.attn.qkv.weight', 
        #                                                                   'blocks.5.norm1.bias', 'blocks.5.norm1.weight', 'blocks.5.norm2.bias', 'blocks.5.norm2.weight', 'blocks.5.mlp.fc1.bias', 'blocks.5.mlp.fc1.weight', 
        #                                                                   'blocks.5.mlp.fc2.bias', 'blocks.5.mlp.fc2.weight', 'blocks.5.attn.proj.bias', 'blocks.5.attn.proj.weight', 'blocks.5.attn.qkv.bias', 'blocks.5.attn.qkv.weight', 
        #                                                                   'blocks.6.norm1.bias', 'blocks.6.norm1.weight', 'blocks.6.norm2.bias', 'blocks.6.norm2.weight', 'blocks.6.mlp.fc1.bias', 'blocks.6.mlp.fc1.weight', 
        #                                                                   'blocks.6.mlp.fc2.bias', 'blocks.6.mlp.fc2.weight', 'blocks.6.attn.proj.bias', 'blocks.6.attn.proj.weight', 'blocks.6.attn.qkv.bias', 'blocks.6.attn.qkv.weight', 
        #                                                                   'blocks.7.norm1.bias', 'blocks.7.norm1.weight', 'blocks.7.norm2.bias', 'blocks.7.norm2.weight', 'blocks.7.mlp.fc1.bias', 'blocks.7.mlp.fc1.weight', 
        #                                                                   'blocks.7.mlp.fc2.bias', 'blocks.7.mlp.fc2.weight', 'blocks.7.attn.proj.bias', 'blocks.7.attn.proj.weight', 'blocks.7.attn.qkv.bias', 'blocks.7.attn.qkv.weight', 
        #                                                                   'blocks.8.norm1.bias', 'blocks.8.norm1.weight', 'blocks.8.norm2.bias', 'blocks.8.norm2.weight', 'blocks.8.mlp.fc1.bias', 'blocks.8.mlp.fc1.weight', 
        #                                                                   'blocks.8.mlp.fc2.bias', 'blocks.8.mlp.fc2.weight', 'blocks.8.attn.proj.bias', 'blocks.8.attn.proj.weight', 'blocks.8.attn.qkv.bias', 'blocks.8.attn.qkv.weight', 
        #                                                                   'blocks.9.norm1.bias', 'blocks.9.norm1.weight', 'blocks.9.norm2.bias', 'blocks.9.norm2.weight', 'blocks.9.mlp.fc1.bias', 'blocks.9.mlp.fc1.weight', 
        #                                                                   'blocks.9.mlp.fc2.bias', 'blocks.9.mlp.fc2.weight', 'blocks.9.attn.proj.bias', 'blocks.9.attn.proj.weight', 'blocks.9.attn.qkv.bias', 'blocks.9.attn.qkv.weight',
        #                                                                   'cross_attn.query.weight', 'cross_attn.key.weight', 'cross_attn.value.weight', 'attn.qkv.weight', 'attn.proj.weight', 'attn.proj.bias', 'norm2.weight', 'norm2.bias'
        #                                                                   ]
        # for k in del_keys:
        #     del weights_dict[k]
        # new_dict = { 'mlp.fc1.weight' if key == 'module.mlp.fc1.weight' else key : value for key, value in weights_dict.items()}
        # new_dict = { 'mlp.fc1.bias' if key == 'module.mlp.fc1.bias' else key : value for key, value in new_dict.items()}
        # new_dict = { 'mlp.fc2.weight' if key == 'module.mlp.fc2.weight' else key : value for key, value in new_dict.items()}
        # new_dict = { 'mlp.fc2.bias' if key == 'module.mlp.fc2.bias' else key : value for key, value in new_dict.items()}
        # new_dict = { 'mlp.fc3.weight' if key == 'module.mlp.fc3.weight' else key : value for key, value in new_dict.items()}
        # new_dict = { 'mlp.fc3.bias' if key == 'module.mlp.fc3.bias' else key : value for key, value in new_dict.items()}
        # new_dict = { 'mlp.fc4.weight' if key == 'module.mlp.fc4.weight' else key : value for key, value in new_dict.items()}
        # new_dict = { 'mlp.fc4.bias' if key == 'module.mlp.fc4.bias' else key : value for key, value in new_dict.items()}
        # new_dict = { 'mlp.norm.weight' if key == 'module.mlp.norm.weight' else key : value for key, value in new_dict.items()}
        # new_dict = { 'mlp.norm.bias' if key == 'module.mlp.norm.bias' else key : value for key, value in new_dict.items()}
        # new_dict = { 'norm1.weight' if key == 'module.norm1.weight' else key : value for key, value in new_dict.items()}
        # weights_dict = { 'norm1.bias' if key == 'module.norm1.bias' else key : value for key, value in new_dict.items()}

        



        print(model.load_state_dict(whole_dict, strict=False))


        # print(teacher1.load_state_dict(teacher1_dict, strict=False))
        # print(teacher2.load_state_dict(teacher2_dict, strict=False))
        # print(teacher3.load_state_dict(teacher3_dict, strict=False))
        # print(teacher4.load_state_dict(teacher4_dict, strict=False))

        

        # new_state_dict = {}
        # for k,v in weights_dict.items():
        #     new_state_dict[k[7:]] = v

        # print(model.load_state_dict(new_state_dict, strict=False))
    else:
        checkpoint_path = os.path.join(
            tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # # 除head, pre_logits外，其他权重全部冻结
            # if "head" not in name and "pre_logits" not in name:
            #     para.requires_grad_(False)
            # else:
            #     print("training {}".format(name))

            if name in ['mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias', 'mlp.fc3.weight', 'mlp.fc3.bias', 'mlp.fc4.weight', 'mlp.fc4.bias', 'mlp.fc5.weight', 'mlp.fc5.bias','mlp.fc6.weight', 'mlp.fc6.bias', 'mlp.norm.weight', 'mlp.norm.bias', 'norm1.weight', 'norm1.bias',
                        'patch_embed_1.proj.weight', 'patch_embed_1.proj.bias','patch_embed_event_1.pos_embedding.weight', 'patch_embed_event_1.pos_embedding.bias', 'patch_embed_event_1.proj.weight', 
                        'patch_embed_event_1.proj.bias']:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                model).to(device)

    # teacher1.eval()
    # teacher2.eval()
    # teacher3.eval()
    # teacher4.eval()
    
    
    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[
                                                      args.gpu], output_device=args.local_rank, find_unused_parameters=True)

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    optimizer = torch.optim.AdamW(pg, lr=args.lr, betas=(
        0.9, 0.999), eps=1e-08, weight_decay=5E-3)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf

    def lf(x): return ((1 + math.cos(x * math.pi / args.epochs)) / 2) * \
        (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train_sampler.set_epoch(epoch)
        # # train
        # train_loss, train_acc_num, train_mlp_loss = train_one_epoch(model=model,
        #                                                             optimizer=optimizer,
        #                                                             data_loader=train_loader,
        #                                                             device=device,
        #                                                             epoch=epoch,
        #                                                             teacher1=teacher1,
        #                                                             teacher2=teacher2,
        #                                                             teacher3=teacher3,
        #                                                             teacher4=teacher4)

        # train_acc = train_acc_num / train_sampler.total_size

        scheduler.step()

        # validate
        val_loss, val_acc_num, val_mlp_loss, val_deg_num, pred_num = evaluate(model=model,
                                                       data_loader=val_loader,
                                                       device=device,
                                                       epoch=epoch)
        val_acc = val_acc_num / val_sampler.total_size
        # val_deg = val_deg_num / val_sampler.total_size

        val_deg = np.mean(val_deg_num)
        val_std = np.std(val_deg_num, ddof=0)


        # print(val_deg_num)
        # print(val_sampler.total_size)

        if rank == 0:
            tags = ["train_loss", "train_acc", "train_mlp_loss",
                    "val_loss", "val_acc", "val_mlp_loss", "learning_rate"]
            # tb_writer.add_scalar(tags[0], train_loss, epoch)
            # tb_writer.add_scalar(tags[1], train_acc, epoch)
            # tb_writer.add_scalar(tags[2], train_mlp_loss, epoch)
            tb_writer.add_scalar(tags[3], val_loss, epoch)
            tb_writer.add_scalar(tags[4], val_acc, epoch)
            tb_writer.add_scalar(tags[5], val_mlp_loss, epoch)

            tb_writer.add_scalar(
                tags[6], optimizer.param_groups[0]["lr"], epoch)
            
            print('valuate-{}:'.format(epoch), 'MAE=', val_deg)
            print('valuate-{}:'.format(epoch), 'STD=', val_std)
            # print('angle_error-{}:'.format(epoch), val_deg_num)
            # count = np.sum(val_deg_num > 1)
            count = sum([1 for i in val_deg_num if i > 10])
            count_1 = len(val_deg_num)
            print("Number of elements greater than 10:", count)
            print("Number of elements:", count_1)

            greaterthan1 = []
            result = []

            for i in range(len(val_deg_num)):
                if val_deg_num[i] > 30:
                    greaterthan1.append(pred_num[i])
                if val_deg_num[i] < 30:
                    result.append(val_deg_num[i])


            element_counts = Counter(greaterthan1)
            print('the frequency of the error position:', element_counts)


            result_MAE = np.mean(result)
            result_std = np.std(result, ddof=0)
            print('valuate-{}:'.format(epoch), 'MAE=', result_MAE)
            print('valuate-{}:'.format(epoch), 'STD=', result_std)



            # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
            # torch.save(model.state_dict(), "./weights/model-current.pth")

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=121)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=False)
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default=os.path.join(os.getcwd(), "all_data/all_data"))
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default=os.path.join(os.getcwd(), "vit_base_patch16_224_in21k.pth"),
    #                     help='initial weights path')
    parser.add_argument('--weights', type=str, default=os.path.join(os.getcwd(), "model-current_0.858.pth"),
                        help='initial weights path')
    # parser.add_argument('--weights1', type=str, default=os.path.join(os.getcwd(), "mlp_weight-1.pth"),
    #                     help='initial weights path')
    # parser.add_argument('--left_top_weights', type=str, default=os.path.join(os.getcwd(), "left_top_weight.pth"),
    #                     help='initial weights path')
    # parser.add_argument('--left_btm_weights', type=str, default=os.path.join(os.getcwd(), "left_btm_weight.pth"),
    #                     help='initial weights path')
    # parser.add_argument('--right_top_weights', type=str, default=os.path.join(os.getcwd(), "right_top_weight.pth"),
    #                     help='initial weights path')
    # parser.add_argument('--right_btm_weights', type=str, default=os.path.join(os.getcwd(), "right_btm_weight.pth"),
    #                     help='initial weights path')

    
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 系统会自动分配
    parser.add_argument('--device', default='cuda',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=-1, type=int)
    opt = parser.parse_args()

    main(opt)
