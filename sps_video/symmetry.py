from os import path
import sys
import math
import random
import time

import torch
import torch.nn.functional as F
import numpy as np

temp_dir = path.abspath(path.join(
    path.dirname(__file__), '../..', 
))
sys.path.append(temp_dir)
from shared import DEVICE
assert sys.path.pop(-1) == temp_dir

ROTATION = 'Rot'
TRANSLATION = 'Trs'


def rand_translation(dim=np.array([1, 0, 1]), is_std_normal=False, t_range=(-1, 1)):
    scale = t_range[1] - t_range[0]
    if is_std_normal:
        s = torch.randn(len(dim))
    else:
        s = torch.rand(len(dim)) * scale + t_range[0]
    s = s.mul(torch.from_numpy(dim))
    s_r = -s
    return s, s_r


#def rate_2d(angle, dim=np.array([1,0,1]))

def rota_Y(angle):
    s = np.array([
        [math.cos(angle), 0, -math.sin(angle)],
        [0, 1, 0],
        [math.sin(angle), 0, math.cos(angle)]
    ])
    s_r = np.array([
        [math.cos(-angle), 0, -math.sin(-angle)],
        [0, 1, 0],
        [math.sin(-angle), 0, math.cos(-angle)]
    ])
    return s, s_r


def rand_rotation_Y(angel_range=(-0.25 * math.pi, 0.25 * math.pi)):
    scale = angel_range[1] - angel_range[0]
    theta = random.random() * scale + angel_range[0]
    s, s_r = rota_Y(theta)
    return torch.from_numpy(s).to(torch.float32), torch.from_numpy(s_r).to(torch.float32), theta

def make_rotation_batch(
    batch_size, r_batch_multiple: int, n_dims: int, 
    angel_range=None, 
):
    # Generalized n-D rotation.

    auged_batch_size = batch_size * r_batch_multiple

    assert n_dims > 1 # 1D rotation not implemented. 
    if n_dims == 2:
        # Use old 2D behaviors. 
        rotate, unrotate, _ = make_rotation_Y_batch(
            auged_batch_size, angel_range, 
        )
        return rotate, unrotate
    # n_dims >= 3
    if angel_range is not None:
        raise ValueError(f'{n_dims}D rotation doesn\'t need `angel_range`.')
    
    # QR can only apply to one matrix. 
    # For speed, we only sample one rotation every "r_multiple". 
    A = torch.randn(
        (r_batch_multiple, n_dims, n_dims), 
        dtype=torch.float32, device=DEVICE, 
    )
    rotates, unrotates = [], []
    for k_i in range(r_batch_multiple):
        rotate  : torch.Tensor = torch.linalg.qr(A[k_i, :, :])[0]
        unrotate: torch.Tensor = torch.linalg.inv(rotate)
        rotates  .append(rotate  .unsqueeze(0).repeat(
            batch_size, 1, 1, 
        ))
        unrotates.append(unrotate.unsqueeze(0).repeat(
            batch_size, 1, 1, 
        ))
    rotates   = torch.cat(rotates,   dim=0)
    unrotates = torch.cat(unrotates, dim=0)
    return rotates, unrotates

def make_rotation_Y_batch(batch_size, angel_range=(-0.25 * math.pi, 0.25 * math.pi)):
    scale = angel_range[1] - angel_range[0]
    theta = torch.rand(batch_size) * scale + angel_range[0]
    s = torch.stack([
        torch.stack([torch.cos(theta), torch.zeros(batch_size), -torch.sin(theta)], dim=0),
        torch.stack([torch.zeros(batch_size), torch.ones(batch_size), torch.zeros(batch_size)], dim=0),
        torch.stack([torch.sin(theta), torch.zeros(batch_size), torch.cos(theta)], dim=0)
    ], dim=0).to(DEVICE).permute(2, 0, 1).contiguous()
    s_r = torch.stack([
        torch.stack([torch.cos(-theta), torch.zeros(batch_size), -torch.sin(-theta)], dim=0),
        torch.stack([torch.zeros(batch_size), torch.ones(batch_size), torch.zeros(batch_size)], dim=0),
        torch.stack([torch.sin(-theta), torch.zeros(batch_size), torch.cos(-theta)], dim=0)
    ], dim=0).to(DEVICE).permute(2, 0, 1).contiguous()
    return s, s_r, theta.numpy()


def make_rotation_2d_batch(batch_size, angel_range=(-0.25 * math.pi, 0.25 * math.pi)):
    scale = angel_range[1] - angel_range[0]
    theta = torch.rand(batch_size) * scale + angel_range[0]
    s = torch.stack([
        torch.stack([torch.cos(theta), -torch.sin(theta)], dim=0),
        torch.stack([torch.sin(theta), torch.cos(theta)], dim=0)
    ], dim=0).to(DEVICE).permute(2, 0, 1).contiguous()
    s_r = torch.stack([
        torch.stack([torch.cos(-theta), -torch.sin(-theta)], dim=0),
        torch.stack([torch.sin(-theta), torch.cos(-theta)], dim=0)
    ], dim=0).to(DEVICE).permute(2, 0, 1).contiguous()
    return s, s_r, theta.numpy()


# def make_translation_batch(
#     auged_batch_size, n_dims, latent_code_num,
#     is_std_normal=False, t_range=(-1, 1),
# ):
#     scale = t_range[1] - t_range[0]
#     if is_std_normal:
#         T_mat = torch.randn(auged_batch_size, n_dims)
#     else:
#         T_mat = torch.rand(auged_batch_size, n_dims) * scale + t_range[0]
#     T = F.pad(T_mat.to(DEVICE), (0, latent_code_num - n_dims, ))
#     T_R = -T
#     return T, T_R


def make_translation_batch(batch_size, dim=np.array([1, 0, 1]), is_std_normal=False, t_range=(-1, 1)):
    scale = t_range[1] - t_range[0]
    if is_std_normal:
        T_mat = torch.randn(batch_size, len(dim))
    else:
        T_mat = torch.rand(batch_size, len(dim)) * scale + t_range[0]
    T = T_mat.mul(torch.from_numpy(dim)).cuda()
    T_R = -T
    return T, T_R


# def make_rotation_Y_batch(batch_size, angel_range=(-0.25 * math.pi, 0.25 * math.pi)):
#     R_batch = []
#     Rr_batch = []
#     theta_batch = []
#     for i in range(batch_size):
#         R, Rr, theta = rand_rotation_Y(angel_range=angel_range)
#         R_batch.append(R)
#         Rr_batch.append(Rr)
#         theta_batch.append(theta)
#     return torch.stack(R_batch, dim=0).to(DEVICE), torch.stack(Rr_batch, dim=0).to(DEVICE), theta_batch


def make_rand_zoom_batch(batch_size, z_range=((0.3, 1.5), (1., 1.), (0.3, 1.5))):
    zoom_range = torch.tensor(z_range)
    zoomer_batch = []
    scale = zoom_range[:, 1] - zoom_range[:, 0]
    min_num = zoom_range[:, 0]
    for i in range(batch_size):
        rand_dim = torch.rand(zoom_range.size(0))
        zoomer = rand_dim * scale + min_num
        zoomer_batch.append(zoomer)
    return torch.stack(zoomer_batch, dim=0).to(DEVICE), 1/torch.stack(zoomer_batch, dim=0).to(DEVICE)


def symm_trans(z, transer):
    return z + transer


def symm_rotaY(z, rotator):
    # z: (batch, 3). rotator: (batch, 3, 3)

    z_R = torch.matmul(z.unsqueeze(1), rotator)
    # This applies linear trans datapoint-wise. 
    return z_R.squeeze(1)

def symm_rota(z: torch.Tensor, rotator: torch.Tensor):
    # Any dim. 
    batch_size, r_dim, _ = rotator.shape
    _, z_dim = z.shape
    assert r_dim == 3
    z_to_rot, z_stay = z.split((r_dim, z_dim - r_dim), dim=1)
    z_rotten = (z_to_rot.unsqueeze(1) @ rotator).squeeze(1)
    return torch.cat((z_rotten, z_stay), dim=1)


def symm_zoom(z, zoomer):
    return z * zoomer


def do_seq_symmetry(z, symm_func):
    z_seq_batch = []
    for i in range(z.size(1)):
        z_S_batch = symm_func(z[:, i])
        z_seq_batch.append(z_S_batch)
    return torch.stack(z_seq_batch, dim=0).permute(1, 0, 2).contiguous()


if __name__ == "__main__":
    t1 = time.time()
    s, s_r, theta = make_rotation_Y_batch(4096, angel_range=(-math.pi, math.pi))
    for i in range(0, 100):
        a = torch.rand(4096, 3).to(DEVICE)
        a_s = symm_rotaY(a, s)
        a_s_sr = symm_rotaY(a_s, s_r)
        print(a - a_s_sr)
    t2 = time.time()
    print(t2 - t1)


    # T, Tr = make_translation_batch(32, t_range=[-3, 3])
    #
    # aaaaa= torch.tensor(((0.3, 1.5), (1., 1.), (0.3, 1.5)))
    # print(aaaaa)
    # zoom, zoom_R = make_rand_zoom_batch(2)
    # print(zoom, zoom_R)
    # z = torch.tensor([[1,2,3],[0,10,100]]).to(DEVICE)
    # zoomed = symm_zoom(z, zoom)
    # print(zoomed)
    # print(symm_zoom(zoomed, zoom_R))

    # t1, t_r1 = rand_translation(is_std_normal=False)
    # print(t1)
    # s1, s_r1, theta = rand_rotation_Y(angel_range=(0, 2 * math.pi))
    # print(f's1: {s1}, s_r1: {s_r1}, mul: {torch.matmul(s1, s_r1)}')
    # a = torch.from_numpy(np.array([
    #     [1.0, 2.0, 3.0]
    # ])).to(torch.float32)
    # a_t = a + t1
    # print(a_t)
    # print(a_t + t_r1)