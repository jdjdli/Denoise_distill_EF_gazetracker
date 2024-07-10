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
user_files = os.listdir(data_path)

for user_index, user_name in enumerate(user_files):
    # if user_name in rest_file:
    eye_list = os.listdir(os.path.join(data_path, user_name))
    for eye_index, eye_num in enumerate(eye_list):

        if not os.path.exists(os.path.join(save_data_path, user_name, eye_num, 'frames_ord1')):
            os.makedirs(os.path.join(save_data_path, user_name, eye_num, 'frames_ord1'))

        load_frame_path = os.path.join(data_path, user_name, eye_num, 'frames_select')

        frame_ord_save_path = os.path.join(save_data_path, user_name, eye_num, 'frames_ord1')
        frames = [os.path.join(load_frame_path, i) for i in os.listdir(load_frame_path)]

        for num, i in enumerate(os.listdir(load_frame_path)):
            img_idx = i.split('_')[0]
            img_row = i.split('_')[1]
            img_col = i.split('_')[2]
            img = cv2.imread(os.path.join(load_frame_path, i))
            cv2.imwrite(os.path.join(frame_ord_save_path,
                                     '%04d' % int(img_idx) + '_' + img_row + '_' + img_col + '.png'), img)



print('all event data processing has been completed!!')
