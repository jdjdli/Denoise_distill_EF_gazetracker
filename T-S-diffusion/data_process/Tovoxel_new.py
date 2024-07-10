import os
import scipy.io
import torch
import numpy as np
import torch.nn.functional as F
import copy

data_path = '../data/all_data'
# event_data_path = os.path.join(os.getcwd(), 'data')
# save_path = os.path.join(os.getcwd(), 'voxel_save')

user_files = os.listdir(data_path)
dvs_img_interval = 1
# rest_file = ['user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user25', 'user26', 'user27']
for user_index, user_name in enumerate(user_files):
    # if user_name in rest_file:
    eye_list = os.listdir(os.path.join(data_path, user_name))
    for eye_index, eye_num in enumerate(eye_list):

        if not os.path.exists(os.path.join(data_path, user_name, eye_num, 'voxels')):
            os.makedirs(os.path.join(data_path, user_name, eye_num, 'voxels'))
        if not os.path.exists(os.path.join(data_path, user_name, eye_num, 'voxels_regular1')):
            os.makedirs(os.path.join(data_path, user_name, eye_num, 'voxels_regular1'))
        # else:
        #     continue
        voxel_save_path = os.path.join(data_path, user_name, eye_num, 'voxels')
        voxel_save_path_regular = os.path.join(data_path, user_name, eye_num, 'voxels_regular1')
        for frame_no in range(len(os.listdir(voxel_save_path))):
                matdata = scipy.io.loadmat(os.path.join(voxel_save_path, 'frame{:0>4d}.mat'.format(frame_no)))
                event_features = np.concatenate((matdata['coor'], matdata['features']), axis=1)
                # event_features = torch.from_numpy(matdata['event_features'])
                if np.isnan(event_features).any():
                    event_features = np.zeros(4096, 19)
                    print('exist nan value in voxel.')

                event_features = torch.from_numpy(event_features)
                z = copy.deepcopy(event_features[:, 0])
                x, y = event_features[:, 1], event_features[:, 2]
                event_features[:, 0] = x
                event_features[:, 1] = y
                event_features[:, 2] = z
                event_features = event_features.unsqueeze(0).unsqueeze(0)
                if event_features.shape[2] < 4096:
                    pad_len = 4096 - event_features.shape[2]
                else:
                    event_features, _ = torch.topk(event_features, k=4096, dim=2)
                    pad_len = 0
                event_features = F.pad(event_features.transpose(-1, -2), (0, pad_len), mode='constant', value=0)
                voxel = event_features.squeeze(0).numpy()

                scipy.io.savemat(os.path.join(voxel_save_path_regular, 'frame{:0>4d}.mat'.format(frame_no)),
                                 mdict={'event_features': voxel})


print('all event data processing has been completed!!')
