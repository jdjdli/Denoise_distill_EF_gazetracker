import os
import scipy.io
import torch
import numpy as np
import torch.nn.functional as F
import copy
import cv2

data_path = '../data/all_data'
# save_data_path = '../data/small_data'
save_data_path = '../data/all_data'
# event_data_path = os.path.join(os.getcwd(), 'data')
# save_path = os.path.join(os.getcwd(), 'voxel_save')

user_files = os.listdir(data_path)
# dvs_img_interval = 1
# rest_file = ['user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user25', 'user26', 'user27']
point_select = [
    (228, 312), (228, 636), (228, 960), (228, 1284), (228, 1608),

    (384, 312), (384, 636), (384, 960), (384, 1284), (384, 1608),

    (540, 312), (540, 636), (540, 960), (540, 1284), (540, 1608),

    (696, 312), (696, 636), (696, 960), (696, 1284), (696, 1608),

    (852, 312), (852, 636), (852, 960), (852, 1284), (852, 1608)

]
for user_index, user_name in enumerate(user_files):
    # if user_name in rest_file:
    eye_list = os.listdir(os.path.join(data_path, user_name))
    for eye_index, eye_num in enumerate(eye_list):

        if not os.path.exists(os.path.join(save_data_path, user_name, eye_num, 'voxels_small')):
            os.makedirs(os.path.join(save_data_path, user_name, eye_num, 'voxels_small'))
        # if not os.path.exists(os.path.join(save_data_path, user_name, eye_num, 'frames_small')):
        #     os.makedirs(os.path.join(save_data_path, user_name, eye_num, 'frames_small'))
        if not os.path.exists(os.path.join(save_data_path, user_name, eye_num, 'frames_small_ord')):
            os.makedirs(os.path.join(save_data_path, user_name, eye_num, 'frames_small_ord'))
        # if not os.path.exists(os.path.join(data_path, user_name, eye_num, 'voxels_regular1')):
        #     os.makedirs(os.path.join(data_path, user_name, eye_num, 'voxels_regular1'))
        # else:
        #     continue
        load_voxel_path = os.path.join(data_path, user_name, eye_num, 'voxels_regular1')
        load_frame_path = os.path.join(data_path, user_name, eye_num, 'frames_select')
        small_voxel_save_path = os.path.join(save_data_path, user_name, eye_num, 'voxels_small')
        # small_frame_save_path = os.path.join(save_data_path, user_name, eye_num, 'frames_small')
        small_frame_ord_save_path = os.path.join(save_data_path, user_name, eye_num, 'frames_small_ord')
        voxels = [os.path.join(load_voxel_path, i) for i in os.listdir(load_voxel_path)]
        frames = [os.path.join(load_frame_path, i) for i in os.listdir(load_frame_path)]
        position = [(int(i.split('_')[1]), int(i.split('_')[2])) for i in os.listdir(load_frame_path)]
        for num, i in enumerate(os.listdir(load_frame_path)):
            if (int(i.split('_')[1]), int(i.split('_')[2])) in point_select:
                img_idx = i.split('_')[0]
                img_row = i.split('_')[1]
                img_col = i.split('_')[2]
                img = cv2.imread(os.path.join(load_frame_path, i))
                cv2.imwrite(os.path.join(small_frame_ord_save_path,
                                         '%04d' % int(img_idx) + '_' + img_row + '_' + img_col + '.png'), img)
                matdata = scipy.io.loadmat(voxels[num])
                event_features = matdata['event_features']
                scipy.io.savemat(os.path.join(small_voxel_save_path, 'frame{:0>4d}.mat'.format(int(img_idx))),
                                 mdict={'event_features': event_features})
                # scipy.io.savemat(voxels[num])



print('all event data processing has been completed!!')
