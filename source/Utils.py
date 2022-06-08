import torch
import numpy as np
import os
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import mrcfile
import torch.nn.functional as F
import math
from hausdorff import hausdorff_distance


# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


def overlap_elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]   #(512,512)表示图像的尺寸
    left_shape = [shape_size[0], shape_size[1] // 2]
    dx_ran = random_state.rand(*shape) * 2 - 1
    dx_ran[:shape_size[0], shape_size[1] // 2:, 0] = (random_state.rand(*left_shape) * 2 - 1) * 0.01
    dx_ran[:, :, 1] = dx_ran[:, :, 0]
    dy_ran = random_state.rand(*shape) * 2 - 1
    dy_ran[:shape_size[0], shape_size[1] // 2:, 0] = (random_state.rand(*left_shape) * 2 - 1) * 0.01
    dy_ran[:, :, 1] = dy_ran[:, :, 0]
    dx = gaussian_filter(dx_ran, sigma) * alpha
    dy = gaussian_filter(dy_ran, sigma) * alpha
    dz = np.zeros_like(dx)  #构造一个尺寸与dx相同的O矩阵
    # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  #网格采样点函数
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), np.stack([dx[..., 0], dy[..., 0]], axis=0)


def overlap_mask_elastic_transform(image, mask, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx_ran_deform = (random_state.rand(*shape) * 2 - 1) * mask
    res_mask = 1 - mask
    dx_ran_res = ((random_state.rand(*shape) * 2 - 1) * 0.01) * res_mask
    dx_ran = dx_ran_deform + dx_ran_res
    dx_ran[:, :, 1] = dx_ran[:, :, 0]
    dy_ran_deform = (random_state.rand(*shape) * 2 - 1) * mask
    dy_ran_res = ((random_state.rand(*shape) * 2 - 1) * 0.01) * res_mask
    dy_ran = dy_ran_deform + dy_ran_res
    dy_ran[:, :, 1] = dy_ran[:, :, 0]
    dx = gaussian_filter(dx_ran, sigma) * alpha
    dy = gaussian_filter(dy_ran, sigma) * alpha
    dz = np.zeros_like(dx)  #构造一个尺寸与dx相同的O矩阵
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  #网格采样点函数
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), np.stack([dx[..., 0], dy[..., 0]], axis=0)


def slight_overlap_elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]   #(512,512)表示图像的尺寸

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1为变换前的坐标，pts2为变换后的坐标，范围为什么是center_square+-square_size？
    # 其中center_square是图像的中心，square_size=512//3=170
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # Mat getAffineTransform(InputArray src, InputArray dst)  src表示输入的三个点，dst表示输出的三个点，获取变换矩阵M
    M = cv2.getAffineTransform(pts1, pts2)  #获取变换矩阵
    #默认使用 双线性插值，
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    left_shape = [shape_size[0], shape_size[1] - shape_size[1] // 10]
    dx_ran = random_state.rand(*shape) * 2 - 1
    dx_ran[:shape_size[0], shape_size[1] // 10:, 0] = (random_state.rand(*left_shape) * 2 - 1) * 0.01
    dx_ran[:, :, 1] = dx_ran[:, :, 0]
    dy_ran = random_state.rand(*shape) * 2 - 1
    dy_ran[:shape_size[0], shape_size[1] // 10:, 0] = (random_state.rand(*left_shape) * 2 - 1) * 0.01
    dy_ran[:, :, 1] = dy_ran[:, :, 0]
    dx = gaussian_filter(dx_ran, sigma) * alpha
    dy = gaussian_filter(dy_ran, sigma) * alpha
    dz = np.zeros_like(dx)  #构造一个尺寸与dx相同的O矩阵
    # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  #网格采样点函数
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), np.stack([dx[..., 0], dy[..., 0]], axis=0)


def rigid_transform_tr(image, rotation, translation, mode='bilinear', inv=False):

    # (512,512)表示图像的尺寸
    shape_size = image.shape
    center = np.float32(shape_size) // 2
    # Random affine
    affine = np.zeros((2, 3))
    s = np.sin(rotation * math.pi / 180.0)
    c = np.cos(rotation * math.pi / 180.0)
    affine[0, 0] = c
    affine[0, 1] = -s
    affine[0, 2] = translation[0]
    affine[1, 0] = s
    affine[1, 1] = c
    affine[1, 2] = translation[1]
    if inv:
        matrix = np.zeros((3, 3))
        matrix[:2, :] = affine
        matrix[2, 2] = 1
        matrix = np.linalg.inv(matrix)
        affine = matrix[:2, :]
    #默认使用 双线性插值，
    # image = cv2.warpAffine(image, warp_m[:2, :], shape_size, borderMode=cv2.BORDER_REFLECT_101)
    if mode == 'bilinear':
        image = cv2.warpAffine(image, affine, shape_size)
    if mode == 'nearest':
        image = cv2.warpAffine(image, affine, shape_size, borderMode=cv2.INTER_NEAREST)
    return image


def standardize_numpy(inputs, constraint=False):
    size = inputs.shape
    if len(size) == 2:
        s = np.std(inputs)
        m = np.mean(inputs)
    if len(size) == 3:
        s = np.std(inputs, axis=(1, 2))
        m = np.mean(inputs, axis=(1, 2))
        m = m.reshape((-1, 1, 1))
        s = s.reshape((-1, 1, 1))
    elif len(size) == 4:
        s = np.std(inputs, axis=(2, 3))
        m = np.mean(inputs, axis=(2, 3))
        m = m.reshape((*size[:2], 1, 1))
        s = s.reshape((*size[:2], 1, 1))

    output = (inputs - m) / s
    if constraint:
        output = np.where(output > -4, output, -4)
        output = np.where(output < 4, output, 4)
    return output


def normalize_numpy(inputs, min_=0, max_=1):
    x_max = np.max(inputs, axis=(1, 2)).reshape(-1, 1, 1)
    x_min = np.min(inputs, axis=(1, 2)).reshape(-1, 1, 1)
    if min_ == 0:
        inputs = (inputs - x_min) / (x_max - x_min)
    if min_ == -1:
        a = 0.5 * (x_max - x_min)
        b = 0.5 * (x_max + x_min)
        inputs = (inputs - b) / a
    return inputs


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    shape [batch, height, width, depth, channel] or [batch, height, width, channel]
    """
    dim = disp.shape[-1]

    if dim == 3:
        D_y = (disp[:, 1:, :-1, :-1, :] - disp[:, :-1, :-1, :-1, :])
        D_x = (disp[:, :-1, 1:, :-1, :] - disp[:, :-1, :-1, :-1, :])
        D_z = (disp[:, :-1, :-1, 1:, :] - disp[:, :-1, :-1, :-1, :])

        D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 1])
        D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
        D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

        return D1 - D2 + D3

    elif dim == 2:
        D_y = (disp[:, 1:, :-1, :] - disp[:, :-1, :-1, :])
        D_x = (disp[:, :-1, 1:, :] - disp[:, :-1, :-1, :])

        D1 = (D_x[..., 0] + 1) * (D_y[..., 1] + 1)
        D2 = D_x[..., 1] * D_y[..., 0]

        return D1 - D2


def patch_mean(images, patch_shape):
    """
    Cal patch mean given patch size
    :param images: (B, C, H, W)
    :param patch_shape: (C, P, P)
    :return:
    """
    channels, *patch_size = patch_shape
    padding = tuple(side // 2 for side in patch_size)
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    channel_selector = torch.eye(channels).byte()
    weights[1 - channel_selector] = 0
    result = F.conv2d(images, weights, padding=padding, bias=None)
    return result


def patch_std(images, patch_shape):
    """
    sqrt E(X^2) - (EX)^2
    """
    return (patch_mean(images**2, patch_shape) - patch_mean(images, patch_shape)**2).sqrt()


def channel_normalize(template):
    """
    Normalize along channel-axis
    """
    rehaped_temp = template.clone().view(template.shape[0], -1)
    rehaped_temp.sub_(rehaped_temp.mean(dim=-1, keepdim=True))
    rehaped_temp.div_(rehaped_temp.std(dim=-1, keepdim=True, unbiased=False))
    return rehaped_temp.view_as(template)


def window_NCC(image, template, patch_shape):
    mole = torch.mean((image - patch_mean(image, patch_shape)) * (template - patch_mean(template, patch_shape)))
    deno = torch.sqrt(torch.mean(torch.square(image - patch_mean(image, patch_shape)))) * \
           torch.sqrt(torch.mean(torch.square(template - patch_mean(template, patch_shape))))
    return mole / deno


def Dice(img1, img2):
    s = []
    row, col = img1.shape
    for r in range(row):
        for c in range(col):
            if img1[r][c] == img2[r][c]:
                s.append(img1[r][c])
    m1 = np.linalg.norm(s)
    m2 = np.linalg.norm(img1.flatten()) + np.linalg.norm(img2.flatten())
    return 2 * m1 / m2


def normalize_img(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def pre_process_nostitch(img1, img2, mask1, mask2):

    mask_added = mask1 + mask2
    mask_super = np.where(mask_added > 0.5, 1.0, 0)
    mask_overlap = np.where(mask_added > 1.5, 1.0, 0)

    return img1, img2, mask1, mask2, mask_super, mask_overlap


def pre_process(img1, img2, mask1, mask2):

    mask_added = mask1 + mask2
    mask_super = np.where(mask_added > 0.5, 1.0, 0)
    mask_overlap = np.where(mask_added > 1.5, 1.0, 0)

    # cut irrelevant region
    raw_pad = img1 + img2 * (mask2 - mask_overlap)
    transformed_pad = img2 + img1 * (mask1 - mask_overlap)
    return raw_pad, transformed_pad, mask1, mask2, mask_super, mask_overlap


def pre_process_cut(img1, img2, mask1, mask2):

    mask_added = mask1 + mask2
    mask_super = np.where(mask_added > 0.5, 1.0, 0)
    mask_overlap = np.where(mask_added > 1.5, 1.0, 0)
    x_map = np.sum(mask_super, axis=1)
    y_map = np.sum(mask_super, axis=0)
    y_l = min(np.argwhere(y_map >= 1.0))[0]
    y_r = max(np.argwhere(y_map >= 1.0))[0]
    x_u = min(np.argwhere(x_map >= 1.0))[0]
    x_d = max(np.argwhere(x_map >= 1.0))[0]

    # cut irrelevant region
    raw_pad = img1 + img2 * (mask2 - mask_overlap)
    raw_pad = raw_pad[x_u: x_d, y_l: y_r]
    transformed_pad = img2 + img1 * (mask1 - mask_overlap)
    transformed_pad = transformed_pad[x_u: x_d, y_l: y_r]
    res_mask1 = mask1[x_u: x_d, y_l: y_r]
    res_mask2 = mask2[x_u: x_d, y_l: y_r]
    res_mask_super = mask_super[x_u: x_d, y_l: y_r]
    res_mask_overlap = mask_overlap[x_u: x_d, y_l: y_r]
    return raw_pad, transformed_pad, res_mask1, res_mask2, res_mask_super, res_mask_overlap


def cut_dark(img):
    x_map = np.sum(img, axis=1)
    y_map = np.sum(img, axis=0)
    y_l = min(np.argwhere(y_map >= 1.0))[0]
    y_r = max(np.argwhere(y_map >= 1.0))[0]
    x_u = min(np.argwhere(x_map >= 1.0))[0]
    x_d = max(np.argwhere(x_map >= 1.0))[0]
    return img[x_u: x_d, y_l: y_r]


def stitch_add_mask(mask1, mask2):
    x_map = np.sum(mask1, axis=1)
    y_map = np.sum(mask1, axis=0)
    y_l = min(np.argwhere(y_map >= 1.0))[0]
    y_r = max(np.argwhere(y_map >= 1.0))[0]
    x_u = min(np.argwhere(x_map >= 1.0))[0]
    x_d = max(np.argwhere(x_map >= 1.0))[0]

    mask_added = mask1 + mask2
    mask_super = np.where(mask_added > 0.5, 1.0, 0)
    mask_overlap = np.where(mask_added > 1.5, 1.0, 0)
    mass = mask_super - mask_overlap
    o_x_map = np.sum(mask_overlap, axis=1)
    o_y_map = np.sum(mask_overlap, axis=0)
    o_y_l = min(np.argwhere(o_y_map >= 1.0))[0]
    o_y_r = max(np.argwhere(o_y_map >= 1.0))[0]
    o_x_u = min(np.argwhere(o_x_map >= 1.0))[0]
    o_x_d = max(np.argwhere(o_x_map >= 1.0))[0]
    x_median = (o_x_u + o_x_d) // 2
    y_median = (o_y_l + o_y_r) // 2
    mass_overlap_1 = mask_overlap / 2
    if abs(o_y_r - y_r) <= 3 and abs(o_x_u - x_u) <= 3:
        mass_overlap_1[x_median:, :y_median] = 0.75
    elif abs(o_y_r - y_r) <= 3 and abs(o_x_d - x_d) <= 3:
        mass_overlap_1[:x_median, :y_median] = 0.75
    elif abs(o_y_l - y_l) <= 3 and abs(o_x_u - x_u) <= 3:
        mass_overlap_1[x_median:, y_median:] = 0.75
    else:
        mass_overlap_1[:x_median, y_median:] = 0.75
    mass_overlap_1 *= mask_overlap
    mass_overlap_2 = (1 - mass_overlap_1) * mask_overlap
    mass_overlap_1 = mass_overlap_1 + mask1 - mask_overlap
    mass_overlap_2 = mass_overlap_2 + mask2 - mask_overlap
    return mass_overlap_1, mass_overlap_2


def stitch_add_mask_linear(mask1, mask2):
    height, width = mask1.shape
    x_map = np.sum(mask1, axis=1)
    y_map = np.sum(mask1, axis=0)
    y_l = min(np.argwhere(y_map >= 1.0))[0]
    y_r = max(np.argwhere(y_map >= 1.0))[0]
    x_u = min(np.argwhere(x_map >= 1.0))[0]
    x_d = max(np.argwhere(x_map >= 1.0))[0]

    mask_added = mask1 + mask2
    mask_super = np.where(mask_added > 0.5, 1.0, 0)
    mask_overlap = np.where(mask_added > 1.5, 1.0, 0)
    mass = mask_super - mask_overlap
    o_x_map = np.sum(mask_overlap, axis=1)
    o_y_map = np.sum(mask_overlap, axis=0)
    o_y_l = min(np.argwhere(o_y_map >= 1.0))[0]
    o_y_r = max(np.argwhere(o_y_map >= 1.0))[0]
    o_x_u = min(np.argwhere(o_x_map >= 1.0))[0]
    o_x_d = max(np.argwhere(o_x_map >= 1.0))[0]
    # x_median = (o_x_u + o_x_d) // 2
    # y_median = (o_y_l + o_y_r) // 2
    mass_overlap_1 = mask_overlap / 2
    tmp_x = np.zeros(mask_overlap.shape)
    tmp_y = np.zeros(mask_overlap.shape)
    if abs(o_y_r - y_r) <= 3 and abs(o_x_u - x_u) <= 3:
        # mass_overlap_1[x_median:, :y_median] = 0.75
        tmp_x[:o_x_d, :] = np.tile(np.linspace(0, 1, o_x_d).reshape(-1, 1), (1, width))
        tmp_y[:, o_y_l:] = np.tile(np.linspace(1, 0, (width - o_y_l)).reshape(1, -1), (height, 1))
        mass_overlap_1 = (tmp_x + tmp_y) / 2
    elif abs(o_y_r - y_r) <= 3 and abs(o_x_d - x_d) <= 3:
        # mass_overlap_1[:x_median, :y_median] = 0.75
        tmp_x[o_x_u:, :] = np.tile(np.linspace(1, 0, (height - o_x_u)).reshape(-1, 1), (1, width))
        tmp_y[:, o_y_l:] = np.tile(np.linspace(1, 0, (width - o_y_l)).reshape(1, -1), (height, 1))
        mass_overlap_1 = (tmp_x + tmp_y) / 2
    elif abs(o_y_l - y_l) <= 3 and abs(o_x_u - x_u) <= 3:
        # mass_overlap_1[x_median:, y_median:] = 0.75
        tmp_x[:o_x_d, :] = np.tile(np.linspace(0, 1, o_x_d).reshape(-1, 1), (1, width))
        tmp_y[:, :o_y_r] = np.tile(np.linspace(0, 1, o_y_r).reshape(1, -1), (height, 1))
        mass_overlap_1 = (tmp_x + tmp_y) / 2
    else:
        # mass_overlap_1[:x_median, y_median:] = 0.75
        tmp_x[o_x_u:, :] = np.tile(np.linspace(1, 0, (height - o_x_u)).reshape(-1, 1), (1, width))
        tmp_y[:, :o_y_r] = np.tile(np.linspace(0, 1, o_y_r).reshape(1, -1), (height, 1))
        mass_overlap_1 = (tmp_x + tmp_y) / 2
    mass_overlap_1 *= mask_overlap
    mass_overlap_2 = (1 - mass_overlap_1) * mask_overlap
    mass_overlap_1 = mass_overlap_1 + mask1 - mask_overlap
    mass_overlap_2 = mass_overlap_2 + mask2 - mask_overlap
    return mass_overlap_1, mass_overlap_2


def stitch_add_mask_linear_border(mask1, mask2, mode=None):
    mask1 = np.where(mask1 > 0.1, 1.0, 0)
    mask2 = np.where(mask2 > 0.1, 1.0, 0)

    height, width = mask1.shape
    x_map = np.sum(mask1, axis=1)
    y_map = np.sum(mask1, axis=0)
    y_l = min(np.argwhere(y_map >= 1.0))[0]
    y_r = max(np.argwhere(y_map >= 1.0))[0]
    x_u = min(np.argwhere(x_map >= 1.0))[0]
    x_d = max(np.argwhere(x_map >= 1.0))[0]

    mask_added = mask1 + mask2
    mask_super = np.where(mask_added > 0.5, 1.0, 0)
    mask_overlap = np.where(mask_added > 1.5, 1.0, 0)
    mass = mask_super - mask_overlap
    o_x_map = np.sum(mask_overlap, axis=1)
    o_y_map = np.sum(mask_overlap, axis=0)
    o_y_l = min(np.argwhere(o_y_map >= 1.0))[0]
    o_y_r = max(np.argwhere(o_y_map >= 1.0))[0]
    o_x_u = min(np.argwhere(o_x_map >= 1.0))[0]
    o_x_d = max(np.argwhere(o_x_map >= 1.0))[0]
    radius_ratio = 0.5

    x_median = (o_x_u + o_x_d) // 2
    x_radius = int((o_x_d - x_median) * 0.25)
    y_median = (o_y_l + o_y_r) // 2
    y_radius = int((o_y_r - y_median) * 0.25)

    # x_median = (o_x_u + o_x_d * 2) // 3
    # x_radius = int((o_x_d - x_median) * 0.1)
    # y_median = (o_y_l * 2 + o_y_r) // 3
    # y_radius = int((o_y_r - y_median) * 0.1)

    mass_overlap_1 = np.zeros(mask_overlap.shape)
    if mode is None:
        if abs(o_x_u - x_u) <= 3:
            # print("u")
            mass_overlap_1[x_u:o_x_d+1, :] = np.tile(np.linspace(0, 1, o_x_d - x_u + 1).reshape(-1, 1), (1, width))
        elif abs(o_x_d - x_d) <= 3:
            # print("d")
            mass_overlap_1[o_x_u:o_x_d+1, :] = np.tile(np.linspace(1, 0, o_x_d - o_x_u + 1).reshape(-1, 1), (1, width))
        elif abs(o_y_l - y_l) <= 3:
            # print("l")
            mass_overlap_1[:, y_l:o_y_r + 1] = np.tile(np.linspace(0, 1, o_y_r - y_l + 1).reshape(1, -1), (height, 1))
        else:
            # print("r")
            mass_overlap_1[:, o_y_l:o_y_r + 1] = np.tile(np.linspace(1, 0, o_y_r - o_y_l + 1).reshape(1, -1), (height, 1))
    else:
        if mode == 'u':
            # mass_overlap_1[x_u:o_x_d + 1, :] = np.tile(np.linspace(0, 1, o_x_d - x_u + 1).reshape(-1, 1), (1, width))
            mass_overlap_1[x_median - x_radius:x_median + x_radius, :] = np.tile(np.linspace(0, 1, 2 * x_radius).reshape(-1, 1), (1, width))
            mass_overlap_1[x_median + x_radius: o_x_d, :] = 1.0
        elif mode == 'd':
            # mass_overlap_1[o_x_u:o_x_d+1, :] = np.tile(np.linspace(1, 0, o_x_d - o_x_u + 1).reshape(-1, 1), (1, width))
            mass_overlap_1[x_median - x_radius:x_median + x_radius, :] = np.tile(np.linspace(1, 0, 2 * x_radius).reshape(-1, 1),
                                                         (1, width))
            mass_overlap_1[o_x_u: x_median - x_radius, :] = 1.0

        elif mode == 'l':
            # mass_overlap_1[:, y_l:o_y_r + 1] = np.tile(np.linspace(0, 1, o_y_r - y_l + 1).reshape(1, -1), (height, 1))
            mass_overlap_1[:, y_median - y_radius:y_median + y_radius] = np.tile(np.linspace(0, 1, 2 * y_radius).reshape(1, -1), (height, 1))
            mass_overlap_1[:, y_median + y_radius: o_y_r] = 1.0
        else:
            # mass_overlap_1[:, o_y_l:o_y_r + 1] = np.tile(np.linspace(1, 0, o_y_r - o_y_l + 1).reshape(1, -1), (height, 1))
            mass_overlap_1[:, y_median - y_radius:y_median + y_radius] = np.tile(np.linspace(1, 0, 2 * y_radius).reshape(1, -1),
                                                         (height, 1))
            mass_overlap_1[:, o_y_l: y_median - y_radius] = 1.0
    mass_overlap_1 *= mask_overlap
    mass_overlap_2 = (1 - mass_overlap_1) * mask_overlap
    mass_overlap_1 = mass_overlap_1 + mask1 - mask_overlap
    mass_overlap_2 = mass_overlap_2 + mask2 - mask_overlap
    return mass_overlap_1, mass_overlap_2


def stitch_add_mask_half(mask1, mask2):
    mask_added = mask1 + mask2
    mass_over = np.where(mask_added > 1.5, 0.5, 0)
    mask1 = mask1 - mass_over
    mask2 = mask2 - mass_over
    return mask1, mask2


def cal_Hausdorff(image1, image2):
    point_1 = np.where(image1 == 255.0, 1, 0)
    point_2 = np.where(image2 == 255.0, 1, 0)
    return hausdorff_distance(point_1, point_2, distance="euclidean")


def psnr(ori, mod):
    diff_square = np.mean(np.square(ori - mod))
    return 10*np.log10(1 / diff_square)


if __name__ == "__main__":
    mask1 = np.zeros((512, 512))
    mask2 = np.zeros((512, 512))
    mask1[200: 400, 200: 400] = 1
    mask2[200: 400, 100: 300] = 1
    mask1, mask2 = stitch_add_mask_linear_border(mask1, mask2, 'l')
    cv2.imshow("mask1", mask1)
    cv2.imshow("mask2", mask2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
