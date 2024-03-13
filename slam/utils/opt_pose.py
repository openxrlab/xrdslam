from copy import deepcopy

import torch
import torch.nn as nn
from pytorch3d.transforms import (matrix_to_quaternion,
                                  quaternion_to_axis_angle,
                                  quaternion_to_matrix)


class OptimizablePose(nn.Module):
    def __init__(self, init_pose, separate_LR=True, rot_rep='axis_angle'):
        super().__init__()
        # init_pose=[tx,ty,tz, qx,qy,qz,qw], init_pose=[tx,ty,tz, rx,ry,rz]
        self.separate_LR = separate_LR
        self.rot_rep = rot_rep
        if separate_LR:
            if rot_rep == 'axis_angle':
                self.register_parameter('data_r', nn.Parameter(
                    init_pose[3:]))  # axis_angle part
                self.register_parameter('data_t', nn.Parameter(
                    init_pose[:3]))  # Translation part
                self.data_r.required_grad_ = True
                self.data_t.required_grad_ = True
            elif rot_rep == 'quat':
                self.register_parameter('data_q', nn.Parameter(
                    init_pose[3:]))  # Quaternion part
                self.register_parameter('data_t', nn.Parameter(
                    init_pose[:3]))  # Translation part
                self.data_q.required_grad_ = True
                self.data_t.required_grad_ = True
            else:
                print('Not support rotation represion: ', rot_rep)
        else:
            self.register_parameter('data', nn.Parameter(init_pose))
            self.data.required_grad_ = True

    def copy_from(self, pose):
        if self.separate_LR:
            if self.rot_rep == 'axis_angle':
                self.data_r = deepcopy(pose.data_r)
                self.data_t = deepcopy(pose.data_t)
            elif self.rot_rep == 'quat':
                self.data_q = deepcopy(pose.data_q)
                self.data_t = deepcopy(pose.data_t)
        else:
            self.data = deepcopy(pose.data)

    def matrix(self):
        Rt = torch.eye(4)
        Rt[:3, :3] = self.rotation()
        Rt[:3, 3] = self.translation()
        return Rt

    def rotation(self):
        if self.rot_rep == 'axis_angle':
            if self.separate_LR:
                rot = self.data_r
            else:
                rot = self.data[3:]
            return self.axis_angle_to_rotation_matrix(rot)
        elif self.rot_rep == 'quat':
            if self.separate_LR:
                q = self.data_q
            else:
                q = self.data[3:]
            return quaternion_to_matrix(q)

    def translation(self):
        if self.separate_LR:
            return self.data_t
        else:
            return self.data[:3]

    @staticmethod
    def axis_angle_to_rotation_matrix(angle_axis):
        angle = torch.norm(angle_axis, dim=-1, keepdim=True)
        axis = angle_axis / angle
        w0, w1, w2 = axis.unbind(dim=-1)
        zeros = torch.zeros_like(w0)
        wx = torch.stack([
            torch.stack([zeros, -w2, w1], dim=-1),
            torch.stack([w2, zeros, -w0], dim=-1),
            torch.stack([-w1, w0, zeros], dim=-1)
        ],
                         dim=-2)
        eye = torch.eye(3, device=angle_axis.device, dtype=angle_axis.dtype)
        R = eye + wx * torch.sin(angle) + (1. - torch.cos(angle)) * (wx @ wx)
        return R

    @classmethod
    def from_matrix(cls, Rt, separate_LR=True, rot_rep='axis_angle'):
        R, u = Rt[:3, :3], Rt[:3, 3]
        quat = matrix_to_quaternion(R)
        axis_angle = quaternion_to_axis_angle(quat)
        if rot_rep == 'axis_angle':
            return OptimizablePose(torch.cat([u, axis_angle], dim=-1),
                                   separate_LR=separate_LR,
                                   rot_rep=rot_rep)  # [tx,ty,tz, rx,ry,rz]
        elif rot_rep == 'quat':
            return OptimizablePose(torch.cat([u, quat], dim=-1),
                                   separate_LR=separate_LR,
                                   rot_rep=rot_rep)


if __name__ == '__main__':
    before = torch.tensor([[-0.955421, 0.119616, -0.269932, 2.655830],
                           [0.295248, 0.388339, -0.872939, 2.981598],
                           [0.000408, -0.913720, -0.406343, 1.368648],
                           [0.000000, 0.000000, 0.000000, 1.000000]])
    pose = OptimizablePose.from_matrix(before, separate_LR=True)
    print(list(pose.parameters()))
    print(len(list(pose.parameters())))
    print(pose.rotation())
    print(pose.translation())
    after = pose.matrix()
    print(after)
    print(torch.abs((before - after)))
