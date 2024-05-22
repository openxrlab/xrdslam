# This file is based on code from Nice-slam,
# (https://github.com/cvg/nice-slam/blob/master/src/utils/datasets.py)
# licensed under the Apache License, Version 2.0.

import csv
import glob
import os

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from slam.common.camera import Camera
from slam.utils.config import load_config
from slam.utils.utils import as_intrinsics_matrix


def readEXR_onlydepth(filename):
    """Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y


def get_dataset(data_path, data_type, device='cuda:0'):
    return dataset_dict[data_type](data_path, device=device)


# RGBD dataset
class BaseDataset(Dataset):
    def __init__(self, data_path, device='cuda:0'):
        super(BaseDataset, self).__init__()
        self.data_format = 'RGBD'

        self.input_folder = data_path
        self.device_yml = os.path.join(data_path, 'devices.yaml')
        cfg = load_config(self.device_yml)
        self.cfg = cfg

        self.device = device
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        # original image intrinsic
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam'][
            'H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg[
                'cam']['cx'], cfg['cam']['cy']
        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_edge = cfg['cam']['crop_edge'] if 'crop_edge' in cfg[
            'cam'] else 0
        self.downsample_factor = cfg['cam'][
            'downsample_factor'] if 'downsample_factor' in cfg['cam'] else 1
        # Camera stores intrinsic after cropping, downsample and de-distortion.
        self.camera = Camera(
            fx=self.fx / self.downsample_factor,
            fy=self.fy / self.downsample_factor,
            cx=(self.cx - self.crop_edge) / self.downsample_factor,
            cy=(self.cy - self.crop_edge) / self.downsample_factor,
            height=int((self.H - 2 * self.crop_edge) / self.downsample_factor),
            width=int((self.W - 2 * self.crop_edge) / self.downsample_factor),
        )

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the
            # color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

        if self.downsample_factor > 1:
            H = (H - 2 * edge) // self.downsample_factor
            W = (W - 2 * edge) // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H),
                                    interpolation=cv2.INTER_LINEAR)
            depth_data = cv2.resize(depth_data, (W, H),
                                    interpolation=cv2.INTER_NEAREST)

        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data)
        pose = self.poses[index]
        return index, color_data.to(self.device).to(
            torch.float), depth_data.to(self.device).to(torch.float), pose.to(
                self.device).to(torch.float)

    def get_camera(self):
        return self.camera


class Replica(BaseDataset):
    def __init__(self, data_path, device='cuda:0'):
        super(Replica, self).__init__(data_path, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')

    def load_poses(self, path):
        self.poses = []
        with open(path, 'r') as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # The codebase assumes that the camera coordinate system is X left
            # to right, Y down to up and Z in the negative viewing direction.
            # Most datasets assume  X left to right, Y up to down and Z in the
            # positive viewing direction. Therefore, we need to rotate the
            # camera coordinate system. Multiplication of R_x (rotation aroun
            # X-axis 180 degrees) from the right.
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


# mono_imu dataset
class Euroc(Dataset):
    def __init__(self, data_path, device='cuda:0'):

        self.data_format = 'MonoImu'

        self.input_folder = data_path
        self.device = device
        self.last_img_timestamp = -1

        self.cam0_yml = os.path.join(data_path, 'mav0/cam0/sensor.yaml')
        cam0_cfg = self.read_yaml(self.cam0_yml)
        self.imu0_yml = os.path.join(data_path, 'mav0/imu0/sensor.yaml')
        imu0_cfg = self.read_yaml(self.imu0_yml)

        # cam0 intrinsics
        self.H, self.W = cam0_cfg['resolution'][1], cam0_cfg['resolution'][0]
        self.fx, self.fy, self.cx, self.cy = cam0_cfg['intrinsics'][
            0], cam0_cfg['intrinsics'][1], cam0_cfg['intrinsics'][2], cam0_cfg[
                'intrinsics'][3]
        self.cam0_distortion = np.array(
            cam0_cfg['distortion_coefficients']
        ) if 'distortion_coefficients' in cam0_cfg else None
        # T_ic  camera to imu
        self.T_ic0 = np.array(cam0_cfg['T_BS']['data']).reshape(4, 4)

        # imu0
        self.gyro_n = imu0_cfg['gyroscope_noise_density']
        self.gyro_rw = imu0_cfg['gyroscope_random_walk']
        self.acc_n = imu0_cfg['accelerometer_noise_density']
        self.acc_rw = imu0_cfg['accelerometer_random_walk']
        self.imu_hz = imu0_cfg['rate_hz']

        self.crop_edge = 0
        self.downsample_factor = 1

        self.camera = Camera(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            height=self.H,
            width=self.W,
        )

        self.img_timestamps, self.color_paths = self.get_imgs_path(
            f'{self.input_folder}/mav0/cam0/data.csv')
        self.n_img = len(self.color_paths)
        self.load_poses(
            f'{self.input_folder}/mav0/state_groundtruth_estimate0/data.csv'
        )  # imu to world
        self.load_imu(f'{self.input_folder}/mav0/imu0/data.csv')

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        color_data = cv2.imread(color_path)

        if self.cam0_distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            color_data_ud = color_data.copy()
            color_data = cv2.undistort(color_data_ud, K, self.cam0_distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        color_data = torch.from_numpy(color_data)  # [H,W,C]

        # pose = self.poses[index]
        pose = self.get_img_pose(self.img_timestamps[index])  # c2w
        imu_datas = torch.empty((0, ))
        if self.last_img_timestamp > 0:
            imu_datas = self.get_imu_data(self.last_img_timestamp,
                                          self.img_timestamps[index])
            imu_datas = torch.from_numpy(imu_datas)
            self.last_img_timestamp = self.img_timestamps[index]

        return index, color_data.to(self.device).to(torch.float), imu_datas.to(
            self.device), pose.to(self.device).to(torch.float)

    def get_img_pose(self, t0):
        nearest_index = np.argmin(np.abs(np.array(self.gt_timestamps) - t0))
        i2w = self.poses[nearest_index]  # imu to world
        c2w = np.dot(i2w, self.T_ic0)  # camera to world, c2w
        # The codebase assumes that the camera coordinate system is X left
        # to right, Y down to up and Z in the negative viewing direction.
        # Most datasets assume  X left to right, Y up to down and Z in the
        # positive viewing direction. Therefore, we need to rotate the
        # camera coordinate system. Multiplication of R_x (rotation aroun
        # X-axis 180 degrees) from the right.
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        return torch.from_numpy(c2w).float()

    def get_imu_data(self, t0, t1):
        out_imus = []
        for i, timestamp in enumerate(self.imu_timestamps):
            if t0 <= timestamp <= t1:
                out_imus.append((timestamp, self.imu_datas[i]))
        return out_imus

    def get_camera(self):
        return self.camera

    def read_yaml(self, path):
        with open(path, 'r') as f:
            first_line = f.readline()
            if first_line.startswith('%'):
                yaml_content = f.read()
            else:
                yaml_content = first_line + f.read()
        yaml_cfg = yaml.safe_load(yaml_content)
        return yaml_cfg

    def get_imgs_path(self, data_csv):
        img_timestamps = []
        color_paths = []
        with open(data_csv, 'r') as file:
            next(file)
            for line in file:
                parts = line.strip().split(',')
                timestamp = int(parts[0])
                filename = parts[1]
                img_timestamps.append(timestamp)
                color_paths.append(
                    os.path.join(os.path.dirname(data_csv), 'data', filename))
        return img_timestamps, color_paths

    def load_imu(self, path):
        self.imu_timestamps = []
        self.imu_datas = []  # t, gyro, acc
        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)
            for row in reader:
                timestamp = int(row[0])
                self.imu_timestamps.append(timestamp)
                imu_data = [float(row[i]) for i in range(1, 7)]
                self.imu_datas.append(imu_data)

    def load_poses(self, path):
        self.gt_timestamps = []
        self.poses = []

        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                self.gt_timestamps.append(int(row[0]))
                line_data = {
                    'position': [float(row[i]) for i in range(1, 4)],
                    'orientation': [float(row[i]) for i in range(4, 8)],
                    'velocity': [float(row[i]) for i in range(8, 11)],
                    'bias_w': [float(row[i]) for i in range(11, 14)],
                    'bias_a': [float(row[i]) for i in range(14, 17)]
                }
                position = np.array(line_data['position'])
                orientation = np.array(line_data['orientation'])
                R = Rotation.from_quat(orientation).as_matrix()
                c2w = np.eye(4)
                c2w[:3, :3] = R
                c2w[:3, 3] = position
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)


class Azure(BaseDataset):
    def __init__(self, data_path, device='cuda:0'):
        super(Azure, self).__init__(data_path, device)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'color', '*.jpg')))
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'depth', '*.png')))
        self.n_img = len(self.color_paths)
        self.load_poses(
            os.path.join(self.input_folder, 'scene', 'trajectory.log'))

    def load_poses(self, path):
        self.poses = []
        if os.path.exists(path):
            with open(path) as f:
                content = f.readlines()

                # Load .log file.
                for i in range(0, len(content), 5):
                    # format %f x 16
                    c2w = np.array(
                        list(
                            map(float, (''.join(
                                content[i + 1:i +
                                        5])).strip().split()))).reshape((4, 4))

                    c2w[:3, 1] *= -1
                    c2w[:3, 2] *= -1
                    c2w = torch.from_numpy(c2w).float()
                    self.poses.append(c2w)
        else:
            for i in range(self.n_img):
                c2w = np.eye(4)
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)


class ScanNet(BaseDataset):
    def __init__(self, data_path, device='cuda:0'):
        super(ScanNet, self).__init__(data_path, device)
        self.input_folder = os.path.join(self.input_folder, 'frames')
        self.color_paths = sorted(glob.glob(
            os.path.join(self.input_folder, 'color', '*.jpg')),
                                  key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(
            os.path.join(self.input_folder, 'depth', '*.png')),
                                  key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
            vals = []
            for line in lines:
                val = list(map(float, line.split(' ')))
                vals.append(val)
            c2w = np.array(vals).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class Scenes7(BaseDataset):
    def __init__(self, data_path, device='cuda:0'):
        super(Scenes7, self).__init__(data_path, device)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, '*.color.png')),
            key=lambda x: int(
                os.path.basename(x).split('.')[0].split('-')[-1]))
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, '*.depth.png')),
            key=lambda x: int(
                os.path.basename(x).split('.')[0].split('-')[-1]))
        self.load_poses(self.input_folder)
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(
            glob.glob(os.path.join(path, '*.txt')),
            key=lambda x: int(
                os.path.basename(x).split('.')[0].split('-')[-1]))
        for pose_path in pose_paths:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
            vals = []
            for line in lines:
                val = list(map(float, line.strip().split('\t')))
                vals.append(val)
            c2w = np.array(vals).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class CoFusion(BaseDataset):
    def __init__(self, data_path, device='cuda:0'):
        super(CoFusion, self).__init__(data_path, device)
        self.input_folder = os.path.join(data_path)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'colour', '*.png')))
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'depth_noise', '*.exr')))
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, 'trajectories'))

    def load_poses(self, path):
        # We tried, but cannot align the coordinate frame of cofusion to ours.
        # So here we provide identity matrix as proxy.
        # But it will not affect the calculation of ATE since camera
        # trajectories can be aligned.
        self.poses = []
        for i in range(self.n_img):
            c2w = np.eye(4)
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, data_path, device='cuda:0'):
        super(TUM_RGBD, self).__init__(data_path, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """read list data."""
        data = np.loadtxt(filepath,
                          delimiter=' ',
                          dtype=np.unicode_,
                          skiprows=skiprows)
        return data

    def associate_frames(self,
                         tstamp_image,
                         tstamp_depth,
                         tstamp_pose,
                         max_dt=0.08):
        """pair images, depths, and poses."""
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """read video data in tum-rgbd format."""
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth,
                                             tstamp_pose)

        indices = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indices[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indices += [i]

        images, poses, depths = [], [], []
        # inv_pose = None
        for ix in indices:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            # if inv_pose is None:
            #     inv_pose = np.linalg.inv(c2w)
            #     c2w = np.eye(4)
            # else:
            #     c2w = inv_pose@c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """convert 4x4 pose matrix to (t, q)"""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


dataset_dict = {
    'replica': Replica,
    'scannet': ScanNet,
    'cofusion': CoFusion,
    'azure': Azure,
    'tumrgbd': TUM_RGBD,
    'euroc': Euroc,
    '7scenes': Scenes7,
}
