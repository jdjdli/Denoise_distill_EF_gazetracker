import os
import numpy as np
import torch
import struct
import glob
from collections import namedtuple
from PIL import Image
from spconv.pytorch.utils import PointToVoxel
import scipy.io
import copy
import torch.nn.functional as F

# from spconv.pytorch.utils import PointToVoxel

'Types of data'
Event = namedtuple('Event', 'polarity row col timestamp')
Frame = namedtuple('Frame', 'row col img timestamp')

'Color scheme for event polarity'
color = ['r', 'g']


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
    return imgs


'Reads an event file'


def read_aerdat(filepath):
    with open(filepath, mode='rb') as file:
        file_content = file.read()

    ''' Packet format'''
    packet_format = 'BHHI'  # pol = uchar, (x,y) = ushort, t = uint32
    packet_size = struct.calcsize('=' + packet_format)  # 16 + 16 + 8 + 32 bits => 2 + 2 + 1 + 4 bytes => 9 bytes
    num_events = len(file_content) // packet_size
    extra_bits = len(file_content) % packet_size

    '''Remove Extra Bits'''
    if extra_bits:
        file_content = file_content[0:-extra_bits]

    ''' Unpacking'''
    event_list = list(struct.unpack('=' + packet_format * num_events, file_content))
    event_list.reverse()

    return event_list


'Parses the filename of the frames'


def get_path_info(path):
    path = path.split('\\')[-1]  # According to windows file path, here double back slashes were used.
    filename = path.split('.')[0]
    path_parts = filename.split('_')
    index = int(path_parts[0])
    stimulus_type = path_parts[3]
    timestamp = int(path_parts[4])
    return {'index': index, 'row': int(path_parts[1]), 'col': int(path_parts[2]), 'stimulus_type': stimulus_type,
            'timestamp': timestamp}


'Manages both events and frames as a general data object'


class EyeDataset:
    'Initialize by creating a time ordered stack of frames and events'

    def __init__(self, data_dir, user):
        self.data_dir = data_dir
        self.user = user

        self.frame_stack = []
        self.event_stack = []

    def __len__(self):
        return len(self.frame_stack) + len(self.event_stack)

    def __getitem__(self, index):
        'Determine if event or frame is next in time by peeking into both stacks'
        frame_timestamp = self.frame_stack[-1].timestamp
        event_timestamp = self.event_stack[-4]

        'Returns selected data type'
        if event_timestamp < frame_timestamp:
            polarity = self.event_stack.pop()
            row = self.event_stack.pop()
            col = self.event_stack.pop()
            timestamp = self.event_stack.pop()
            event = Event(polarity, row, col, timestamp)
            return event
        else:
            frame = self.frame_stack.pop()
            img = Image.open(frame.img).convert("L")
            frame = frame._replace(img=img)
            return frame

    'Loads in data from the data_dir as filenames'

    def collect_data(self, eye=0):
        print('Loading Frames....')
        self.frame_stack = self.load_frame_data(eye)
        print('There are ' + str(len(self.frame_stack)) + ' frames \n')
        print('Loading Events....')
        self.event_stack = self.load_event_data(eye)
        print('There are ' + str(len(self.event_stack)) + ' events \n')

    def load_frame_data(self, eye):
        filepath_list = []
        user_name = str(self.user)
        img_dir = os.path.join(self.data_dir, user_name, str(eye), 'frames_select')
        img_filepaths = list(glob_imgs(img_dir))
        img_filepaths.sort(key=lambda name: get_path_info(name)['index'])
        img_filepaths.reverse()
        for fpath in img_filepaths:
            path_info = get_path_info(fpath)
            frame = Frame(path_info['row'], path_info['col'], fpath, path_info['timestamp'])
            filepath_list.append(frame)
        return filepath_list

    def load_event_data(self, eye):
        user_name = str(self.user)
        event_file = os.path.join(self.data_dir, user_name, str(eye), 'events.aerdat')
        filepath_list = read_aerdat(event_file)
        return filepath_list


def transform_eventpoints_to_voxels(data_dict={}, voxel_generator=None, device=torch.device("cuda:0")):
    """
    将event points转换为voxel,调用spconv的VoxelGeneratorV2
    """
    points = data_dict['points']
    # 将points打乱
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]
    data_dict['points'] = points

    # 使用spconv生成voxel输出
    points = torch.as_tensor(data_dict['points']).to(device)
    voxel_output = voxel_generator(points)

    # 假设一份点云数据是N*4，那么经过pillar生成后会得到三份数据
    # voxels代表了每个生成的voxel数据，维度是[voxel_num, max_num_points_per_voxel, 4]
    # coordinates代表了每个生成的voxel所在的zyx轴坐标，维度是[max_num_points_per_voxel,3]
    # num_points代表了每个生成的voxel中有多少个有效的点维度是[voxel_num,]，因为不满max_point会被0填充
    voxels, coordinates, num_points = voxel_output
    voxels = voxels.to(device)
    coordinates = coordinates.to(device)
    num_points = num_points.to(device)
    # 选event数量在前5000的voxel  8000 from(4k+,6k+)
    # print(torch.where(num_points>=16)[0].shape)
    if num_points.shape[0] < save_voxel:
        features = voxels[:, :, 3]
        coor = coordinates[:, :]
    else:
        _, voxels_idx = torch.topk(num_points, save_voxel)
        # 将每个voxel的1024个p拼接作为voxel初始特征   16
        features = voxels[voxels_idx][:, :, 3]
        # 前5000个voxel的三维坐标
        coor = coordinates[voxels_idx]
    # 将y.x.t改为t,x,y
    coor[:, [0, 1, 2]] = coor[:, [2, 1, 0]]

    return coor, features


if __name__ == '__main__':

    save_voxel = 5000
    device = torch.device("cuda:0")
    voxel_generator = PointToVoxel(
        # 给定每个voxel的长宽高  [0.05, 0.05, 0.1]
        vsize_xyz=[50, 10, 10],  # [0.2, 0.25, 0.16]  # [50, 10, 10]  [50, 35, 26] 因此坐标范围（20,20,20）  (20, 34/35, 26)
        # 给定点云的范围
        coors_range_xyz=[0, 0, 0, 1000, 345, 259],
        # 给定每个点云的特征维度，这里是x，y，z，r 其中r是激光雷达反射强度       # 346x260  t,x,y
        num_point_features=4,
        # 最多选取多少个voxel，训练16000，推理40000
        max_num_voxels=16000,  # 16000
        # 给定每个pillar中有采样多少个点，不够则补0  因此我将neg voxel改为-1;
        max_num_points_per_voxel=16,  # 1024
        device=device
    )

    # data_path = os.path.join(os.getcwd(), '/data/all_data')
    data_path = '../data/all_data'
    # event_data_path = os.path.join(os.getcwd(), 'data')
    # save_path = os.path.join(os.getcwd(), 'voxel_save')

    user_files = os.listdir(data_path)
    dvs_img_interval = 1
    rest_file = ['user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'user25', 'user26', 'user27']
    for user_index, user_name in enumerate(user_files):
        if user_name in rest_file:
            eye_list = os.listdir(os.path.join(data_path, user_name))
            for eye_index, eye_num in enumerate(eye_list):

                if not os.path.exists(os.path.join(data_path, user_name, eye_num, 'voxels')):
                    os.makedirs(os.path.join(data_path, user_name, eye_num, 'voxels'))
                if not os.path.exists(os.path.join(data_path, user_name, eye_num, 'voxels_regular')):
                    os.makedirs(os.path.join(data_path, user_name, eye_num, 'voxels_regular'))
                # else:
                #     continue
                voxel_save_path = os.path.join(data_path, user_name, eye_num, 'voxels')
                voxel_save_path_regular = os.path.join(data_path, user_name, eye_num, 'voxels_regular')

                # event_file = 'events.aerdat'
                # event_data_path = os.path.join(data_path, user_name, eye_num, event_file)
                'Manages both events and frames as a general data object'
                eye_dataset = EyeDataset(data_path, user_name)
                if eye_num == '0':
                    print('Showing the left eye of subject ' + str(user_name) + '\n')
                    print('Loading Data from ' + data_path + '..... \n')
                    eye_dataset.collect_data(0)
                else:
                    print('Showing the right eye of subject ' + str(user_name) + '\n')
                    print('Loading Data from ' + data_path + '..... \n')
                    eye_dataset.collect_data(1)

                frame_all = []
                frame_timestamp = []
                event_col = []
                event_row = []
                event_polarity = []
                event_timestamp = []
                for i, data in enumerate(eye_dataset):
                    if type(data) is Frame:
                        frame_all.append(data.img)
                        frame_timestamp.append(data.timestamp)
                    else:
                        event_col.append(data.col)
                        event_row.append(data.row)
                        event_polarity.append(data.polarity)
                        event_timestamp.append(data.timestamp)
                frame_num = len(frame_timestamp)
                frame_t_all = torch.tensor(frame_timestamp).unsqueeze(1).to(device)
                t_all = torch.tensor(event_timestamp).unsqueeze(1).to(device)
                x_all = torch.tensor(event_row).unsqueeze(1).to(device)
                y_all = torch.tensor(event_col).unsqueeze(1).to(device)
                p_all = torch.tensor(event_polarity).unsqueeze(1).to(device)
                for frame_no in range(0, int(frame_num) - 2):
                    start_idx = \
                        [i for i, timestamp in enumerate(event_timestamp) if timestamp > frame_timestamp[frame_no]][
                            0]
                    end_idx = \
                        [i for i, timestamp in enumerate(event_timestamp) if timestamp > frame_timestamp[frame_no + 1]][
                            0]
                    # start_idx = np.where(event_timestamp >= frame_timestamp)[0][0]
                    # end_idx = np.where(event_timestamp >= frame_timestamp[frame_no + 1])[0][0]

                    # if frame_no == 709:
                    #     aaa = 0

                    t = t_all[start_idx:end_idx]
                    if start_idx == end_idx:

                        time_length = 0
                        t = ((t - t).float() / time_length) * 1000
                        scipy.io.savemat(os.path.join(voxel_save_path, 'frame{:0>4d}.mat'.format(frame_no)),
                                         mdict={'coor': np.zeros([100, 3]),
                                                'features': np.zeros(
                                                    [100, 16])})  # coor: numpy.nan;   features: numpy.nan

                        event_features = np.zeros([4096, 19])
                        print('exist nan value in voxel.')
                        event_features = torch.from_numpy(event_features)
                        event_features = event_features.unsqueeze(0)
                        event_features = event_features.transpose(-1, -2).numpy()

                        scipy.io.savemat(os.path.join(voxel_save_path_regular, 'frame{:0>4d}.mat'.format(frame_no)),
                                         mdict={'event_features': event_features})
                        print('empty event frame ', frame_no)
                        continue
                    else:
                        time_length = t[-1] - t[0]
                        # rescale the timestampes to start from 0 up to 1000
                        t = ((t - t[0]).float() / time_length) * 1000
                        # t = t[start_idx:end_idx]
                        x = x_all[start_idx:end_idx]
                        y = y_all[start_idx:end_idx]
                        p = p_all[start_idx:end_idx]
                        current_events = torch.cat((t, x, y, p), dim=1)

                        data_dict = {'points': current_events}
                        try:
                            coor, features = transform_eventpoints_to_voxels(data_dict=data_dict,
                                                                             voxel_generator=voxel_generator,
                                                                             device=device)
                            # pdb.set_trace()
                            coor = coor.cpu().numpy()
                            features = features.cpu().numpy()
                            # print('coor', coor)
                            scipy.io.savemat(os.path.join(voxel_save_path, 'frame{:0>4d}.mat'.format(frame_no)),
                                             mdict={'coor': coor,
                                                    'features': features})  # coor: Nx(t,x,y);   features:Nx32 or Nx10024
                            event_features = np.concatenate((coor, features),
                                                            axis=1)  # concat coorelate and features (x,y,z, feauture32/16)
                            if np.isnan(event_features).any():
                                event_features = np.zeros([4096, 19])
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
                            event_features = event_features.squeeze(0).numpy()
                            scipy.io.savemat(os.path.join(voxel_save_path_regular, 'frame{:0>4d}.mat'.format(frame_no)),
                                             mdict={'event_features': event_features})
                            # print('empty event frame ', frame_no)
                        except:
                            scipy.io.savemat(os.path.join(voxel_save_path, 'frame{:0>4d}.mat'.format(frame_no)),
                                             mdict={'coor': np.zeros([100, 3]),
                                                    'features': np.zeros(
                                                        [100, 16])})  # coor: numpy.nan;   features: numpy.nan
                            event_features = np.zeros([4096, 19])
                            print('exist nan value in voxel.')
                            event_features = torch.from_numpy(event_features)
                            event_features = event_features.unsqueeze(0)
                            event_features = event_features.transpose(-1, -2).numpy()

                            scipy.io.savemat(os.path.join(voxel_save_path_regular, 'frame{:0>4d}.mat'.format(frame_no)),
                                             mdict={'event_features': event_features})
                            print('problem event frame ', frame_no)

            else:
                continue
    print('all event data processing has been completed!!')
