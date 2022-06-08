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
from Utils import normalize_img, overlap_mask_elastic_transform, rigid_transform_tr, window_NCC
from tqdm import tqdm
import random


# def cut_overlap(mask1, mask2):
#     mask_added = mask1 + mask2
#     mask_overlap = np.where(mask_added > 1.5, 1.0, 0)
#
#     mask1 = mask1 - mask_overlap
#     mask2 = mask2 - mask_overlap
#     return mask1, mask2
#
#
# data_dir = "../data"
# test_dir = 'test/10'
# raw_path = "%s/%s/raw_left.png" % (data_dir, test_dir)
# deformed_path = "%s/%s/deformed_right.png" % (data_dir, test_dir)
# deformed_mask = "%s/%s/mask_right.png" % (data_dir, test_dir)
#
# raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
# transformed_img = cv2.imread(deformed_path, cv2.IMREAD_GRAYSCALE)
# transformed_mask_img = cv2.imread(deformed_mask, cv2.IMREAD_GRAYSCALE)
# transformed_mask_img = np.where(transformed_mask_img > 0.1, 1.0, 0)
# raw_mask_img = np.zeros(raw_img.shape)
# raw_mask_img[raw_img > 0.1] = 1
#
# # raw_mask_img = np.stack([np.zeros((1024, 1024)), np.zeros((1024, 1024)), raw_mask_img], axis=2)
# # transformed_mask_img = np.stack([transformed_mask_img, np.zeros((1024, 1024)), np.zeros((1024, 1024))], axis=2)
#
# raw_mask_img_cut, transformed_mask_img_cut = cut_overlap(raw_mask_img, transformed_mask_img)
#
# cv2.imshow("transformed_mask", transformed_mask_img)
# cv2.imshow("raw_mask", raw_mask_img)
# cv2.imshow("transformed_mask_cut", transformed_mask_img_cut)
# cv2.imshow("raw_mask_cut", raw_mask_img_cut)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
cut_dark
"""
# data_dir = "../data/test/22"
# mask1 = cv2.imread("%s/mask_left.png" % data_dir, cv2.IMREAD_GRAYSCALE)
# mask1 = np.where(mask1 > 0.1, 1.0, 0)
# mask2 = cv2.imread("%s/mask_right.png" % data_dir, cv2.IMREAD_GRAYSCALE)
# mask2 = np.where(mask2 > 0.1, 1.0, 0)
# mask_super = mask1 + mask2
# x_map = np.sum(mask_super, axis=1)
# y_map = np.sum(mask_super, axis=0)
# y_l = min(np.argwhere(y_map >= 1.0))[0]
# y_r = max(np.argwhere(y_map >= 1.0))[0]
# x_u = min(np.argwhere(x_map >= 1.0))[0]
# x_d = max(np.argwhere(x_map >= 1.0))[0]
# mask1 = mask1[x_u: x_d, y_l: y_r]
# mask2 = mask2[x_u: x_d, y_l: y_r]
# # cv2.imshow("mask1_cut", mask1)
# # cv2.imshow("mask2_cut", mask2)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# for root, dirs, files in os.walk(data_dir):
#     for file in files:
#         # print(file)
#         img = cv2.imread("%s/%s" % (data_dir, file), cv2.IMREAD_GRAYSCALE)
#         img = img[x_u: x_d, y_l: y_r]
#         cv2.imwrite("%s/cut/%s" % (data_dir, file), img)

# file = "warp_right_white.png"
# img = cv2.imread("%s/%s" % (data_dir, file), cv2.IMREAD_GRAYSCALE)
# img = img[x_u: x_d, y_l: y_r]
# cv2.imwrite("%s/cut/%s" % (data_dir, file), img)

# file = "warp_left_APAP.png"
# img = cv2.imread("%s/%s" % (data_dir, file), -1)
# img = img[x_u: x_d, y_l: y_r, :]
# cv2.imwrite("%s/cut/%s" % (data_dir, file), img)

# data_dir = "../data/test/10"
# mask = cv2.imread("%s/mask_left.png" % data_dir, cv2.IMREAD_GRAYSCALE)
# feature_1 = cv2.imread("%s/1_left.png" % data_dir, cv2.IMREAD_GRAYSCALE)
# feature_2 = cv2.imread("%s/1_right.png" % data_dir, cv2.IMREAD_GRAYSCALE)
# mask = cv2.resize(mask, (512, 512))
# mask = np.where(mask > 0.01, 1.0, 0)
# feature_1 = feature_1 * mask
# feature_2 = feature_2 * mask
# cv2.imshow("1", np.uint8(feature_1))
# cv2.imshow("2", np.uint8(feature_2))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# data_dir = "../data/test/real_3"
# for root, dirs, files in os.walk(data_dir):
#     for file in files:
#         # print(file)
#         img = cv2.imread("%s/%s" % (data_dir, file), cv2.IMREAD_GRAYSCALE)
#         x_map = np.sum(img, axis=1)
#         y_map = np.sum(img, axis=0)
#         y_l = min(np.argwhere(y_map >= 1.0))[0]
#         y_r = max(np.argwhere(y_map >= 1.0))[0]
#         x_u = min(np.argwhere(x_map >= 1.0))[0]
#         x_d = max(np.argwhere(x_map >= 1.0))[0]
#         img = img[x_u: x_d, y_l: y_r]
#         cv2.imwrite("%s/cut/%s" % (data_dir, file), img)

"""
clarity
"""
# data_dir = "../data/test/22"
# test_dir = "cut"
# # method = "our"
# # method = "uni"
# # method = "unet"
# # method = "REW"
# # method = "autostitch"
# method = "APAP"
# # rrr = cv2.imread("%s/%s/warp_left_%s.png" % (data_dir, test_dir, method))
# rrr = cv2.imread("%s/%s/warp_right_%s.png" % (data_dir, test_dir, method))
# # lll = cv2.cvtColor(lll, cv2.COLOR_BGR2BGRA)
# rrr = cv2.cvtColor(rrr, cv2.COLOR_BGR2BGRA)
# # w, h, c = lll.shape
# w, h, c = rrr.shape
# for i in range(w):
#     for j in range(h):
#         # b, g, r, _ = lll[i, j]
#         # if r == 0 and g == 0 and b == 0:
#         #     lll[i][j] = [0, 0, 0, 0]
#         b, g, r, _ = rrr[i, j]
#         if r == 0 and g == 0 and b == 0:
#             rrr[i][j] = [0, 0, 0, 0]
# # cv2.imwrite("%s/%s/warp_left_%s.png" % (data_dir, test_dir, method), rrr)
# cv2.imwrite("%s/%s/warp_right_%s.png" % (data_dir, test_dir, method), rrr)


"""
real img process
"""
# data_path = "../data/333/3/temp_2"
# left_path = "%s/left_img.png" % data_path
# right_path = "%s/right_img.png" % data_path
# mask_1_path = "%s/left_mask.png" % data_path
# mask_2_path = "%s/right_mask.png" % data_path
# left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
# right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
# left_mask_img = cv2.imread(mask_1_path, cv2.IMREAD_GRAYSCALE)
# right_mask_img = cv2.imread(mask_2_path, cv2.IMREAD_GRAYSCALE)
# panorama = 2560
# offset_h = 500
# offset_w = 500
# raw_left = np.zeros((panorama, panorama))
# h, w = left_img.shape
# raw_left[offset_h: offset_h + h, offset_w: offset_w + w] = left_img
# mask_left = np.zeros((panorama, panorama))
# mask_left[offset_h: offset_h + h, offset_w: offset_w + w] = left_mask_img
# deformed_right = np.zeros((panorama, panorama))
# h, w = right_img.shape
# deformed_right[offset_h: offset_h + h, offset_w: offset_w + w] = right_img
# mask_right = np.zeros((panorama, panorama))
# mask_right[offset_h: offset_h + h, offset_w: offset_w + w] = right_mask_img
# new_left_path = "%s/raw_left.png" % data_path
# new_right_path = "%s/deformed_right.png" % data_path
# new_mask_left_path = "%s/mask_left.png" % data_path
# new_mask_right_path = "%s/mask_right.png" % data_path
# cv2.imwrite(new_left_path, raw_left)
# cv2.imwrite(new_right_path, deformed_right)
# cv2.imwrite(new_mask_left_path, mask_left)
# cv2.imwrite(new_mask_right_path, mask_right)


# data_dir = "../data/333/3/temp"
# file = "stitching_result"
# # file = "mask_super"
# img = cv2.imread("%s/%s.png" % (data_dir, file), cv2.IMREAD_GRAYSCALE)
# x_map = np.sum(img, axis=1)
# y_map = np.sum(img, axis=0)
# y_l = min(np.argwhere(y_map >= 1.0))[0]
# y_r = max(np.argwhere(y_map >= 1.0))[0]
# x_u = min(np.argwhere(x_map >= 1.0))[0]
# x_d = max(np.argwhere(x_map >= 1.0))[0]
# img = img[x_u: x_d, y_l: y_r]
# cv2.imwrite("%s/%s_cut.png" % (data_dir, file), img)


data_path = "../data/333/3/temp_2"
left_path = "%s/left_img.png" % data_path
right_path = "%s/right_img.png" % data_path
mask_1_path = "%s/left_mask.png" % data_path
mask_2_path = "%s/right_mask.png" % data_path
left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
left_img = torch.Tensor(left_img).float()
right_img = torch.Tensor(right_img).float()
size = left_img.shape
left_img = left_img.view((-1, 1, *size))
right_img = right_img.view((-1, 1, *size))
x_u = 260
x_d = x_u + 58
y_l = 929
y_r = y_l + 37
print(window_NCC(left_img[:, :, x_u: x_d, y_l: y_r], right_img[:, :, x_u: x_d, y_l: y_r], (1, 35, 35)).data)

"""
different deformation
"""
# # path = "../data/sample_C_padded_20160501.mrc"
# path = "../data/sample_C_20160501_1024.mrc"
# with mrcfile.open(path) as mrc:
#     data = mrc.data
#
# print(data.shape)
# rest_index = []
# for i in range(len(data)):
#     # if i+1 not in [52, 112, 124]:
#     if i + 1 not in [15, 75, 87, 116]:
#         rest_index.append(i)
# # data = data[rest_index]
# data = data[rest_index, 113: -113, 113: -113]
# print(data.shape)
#
# seg = np.load("../data/sample_C_20160501_edge_1024.npy")
# seg = seg[rest_index, 113: -113, 113: -113]
# print(seg.shape)
#
# i = 20
# k = 7
# total_img = normalize_img(data[i, :, :])
# total_seg = seg[i, :, :]
# raw = np.zeros((1024, 1024))
# raw[300: 812, 300: 812] = total_img[300: 812, 300: 812]
# raw_seg = np.zeros((1024, 1024))
# raw_seg[300: 812, 300: 812] = total_seg[300: 812, 300: 812]
#
# rotation = -5.0
# translation = [-70.0, -100.0]
#
# transformed_total = rigid_transform_tr(total_img, rotation, translation)
# transformed_total = normalize_img(transformed_total)
# transformed_total_seg = rigid_transform_tr(total_seg, rotation, translation)
#
# mask = np.zeros((1024, 1024))
# mask[300: 812, 300: 812] = 1
# transformed_mask = rigid_transform_tr(mask, rotation, translation, 'nearest', inv=True)
# cv2.imshow("mask_right", transformed_mask)
# #
# # mask_added = mask + transformed_mask
# # cv2.imshow("mask_added", mask_added)
#
# transformed_overlap = rigid_transform_tr(mask, rotation, translation, 'nearest')
# raw_mask = np.zeros((1024, 1024))
# raw_mask[300: 812, 300: 812] = 1
# transformed_overlap = transformed_overlap + raw_mask
# # cv2.imshow("raw + trans", transformed_overlap)
# transformed_overlap[transformed_overlap < 1.5] = 0
# transformed_overlap[transformed_overlap > 1.5] = 1
# # deformed_mask = np.zeros(transformed_total.shape)
# deformed_mask = transformed_total_seg
# im_merge = np.concatenate((transformed_total[..., None], deformed_mask[..., None]), axis=2)
# im_trans_mask = np.concatenate((transformed_overlap[..., None], transformed_overlap[..., None]), axis=2)
# im_t = None
# im_seg_t = None
# # indices = None
# count = 0
# while count < k:
#     im_merge_t, indices = overlap_mask_elastic_transform(im_merge, im_trans_mask, im_merge.shape[1] * 2,
#                                                          im_merge.shape[1] * 0.08,
#                                                          random_state=np.random.RandomState(135549))
#     im_t = im_merge_t[..., 0]
#     im_seg_t = im_merge_t[..., 1]
#     count += 1
#     transformed_right = im_t * mask
#     transformed_right = rigid_transform_tr(transformed_right, rotation, translation, 'bilinear', inv=True)
#     cv2.imshow("deformed_right_%d" % count, transformed_right)
#     im_merge = im_merge_t
#
# # cv2.imshow("deformed_right", transformed_right)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
