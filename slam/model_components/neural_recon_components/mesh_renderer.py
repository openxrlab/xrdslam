# This file is based on code from NeuralRecon,
# licensed under the Apache License, Version 2.0.

import os

import numpy as np
import pyrender

os.environ['PYOPENGL_PLATFORM'] = 'egl'


class Renderer():
    """OpenGL mesh renderer.

    Used to render depthmaps from a mesh for 2d evaluation
    """
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2],
                                        cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0],
                                        fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
