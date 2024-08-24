""" Implementation from https://github.com/qinzheng93/Easy-KPConv/blob/master/setup.py. """
import os.path as osp
import numpy as np

from os import makedirs
from os.path import join, exists
from mmdetection3d.projects.KPPillarsBEV.kppillarsbev.kpconv.kernel_points import kernel_point_optimization_debug


def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """
    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack(
        [t1 + t2 * t3, t7 - t9, t11 + t12, t7 + t9, t1 + t2 * t15, t19 - t20, t11 - t12, t19 + t20, t1 + t2 * t24],
        axis=1,
    )

    return np.reshape(R, (-1, 3, 3))


def load_kernels(radius, num_kpoints, dimension, fixed, lloyd=False):
    # Kernel directory
    kernel_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'dispositions')
    if not exists(kernel_dir):
        makedirs(kernel_dir)

    # # To many points switch to Lloyds
    # if num_kpoints > 30:
    #     lloyd = True

    # Kernel_file
    kernel_file = join(kernel_dir, 'k_{:03d}_{:s}_{:d}D.npy'.format(num_kpoints, fixed, dimension))

    # Check if already done
    if not exists(kernel_file):
        # Create kernels
        kernel_points, grad_norms = kernel_point_optimization_debug(
            1.0, num_kpoints, num_kernels=100, dimension=dimension, fixed=fixed, verbose=0
        )
        # Find best candidate
        best_k = np.argmin(grad_norms[-1, :])

        # Save points
        kernel_points = kernel_points[best_k, :]
        np.save(kernel_file, kernel_points)
    else:
        kernel_points = np.load(kernel_file)

    # Random rotations for the kernel
    # N.B. 4D random rotations not supported yet
    R = np.eye(dimension)
    theta = np.random.rand() * 2 * np.pi
    if dimension == 2:
        if fixed != 'vertical':
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
    elif dimension == 3:
        if fixed != 'vertical':
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        else:
            phi = (np.random.rand() - 0.5) * np.pi
            # Create the first vector in carthesian coordinates
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi
            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]
            R = R.astype(np.float32)

    # Add a small noise
    kernel_points = kernel_points + np.random.normal(scale=0.01, size=kernel_points.shape)
    # Scale kernels
    kernel_points = radius * kernel_points
    # Rotate kernels
    kernel_points = np.matmul(kernel_points, R)

    return kernel_points.astype(np.float32)
