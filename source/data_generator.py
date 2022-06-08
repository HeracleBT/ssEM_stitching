import mrcfile
import numpy as np
import cv2
import torch
from layers import *
import torch.nn.functional as F
import os
import random
from tqdm import tqdm
from Utils import overlap_elastic_transform, overlap_mask_elastic_transform, normalize_img, rigid_transform_tr, \
    standardize_numpy, pre_process_cut


"""
train data generator
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
#
# index = 10
# total_img = data[10]
# total_img = normalize_img(total_img)
# raw_left = total_img[300: 812, 300: 812]
#
# # rotation = random.uniform(5.0, 15.0) * random.choice([-1, 1])
# # translation = [random.uniform(100.0, 250.0) * random.choice([-1, 1]), random.uniform(100.0, 250.0) * random.choice([-1, 1])]
# rotation = 10
# translation = [-150, -150]
#
# transformed_total = rigid_transform_tr(total_img, rotation, translation)
# transformed_total = normalize_img(transformed_total)
# raw_right = np.zeros((1024, 1024))
# raw_right[300: 812, 300: 812] = transformed_total[300: 812, 300: 812]
#
# mask = np.zeros((1024, 1024))
# mask[300: 812, 300: 812] = 1
# transformed_mask = rigid_transform_tr(mask, rotation, translation, 'nearest', inv=True)
#
# # transformed_right = rigid_transform_tr(raw_right, rotation, translation, 'nearest', inv=True)
# #
# # mask_super = transformed_mask.copy()
# # mask_super[300: 812, 300: 812] = 1
# # transformed_super = transformed_right.copy()
# # transformed_super[300: 812, 300: 812] = raw_left
# #
# # x_map = np.sum(mask_super, axis=1)
# # y_map = np.sum(mask_super, axis=0)
# # y_l = min(np.argwhere(y_map >= 1.0))[0]
# # y_r = max(np.argwhere(y_map >= 1.0))[0]
# # x_u = min(np.argwhere(x_map >= 1.0))[0]
# # x_d = max(np.argwhere(x_map >= 1.0))[0]
# # # print(y_l)
# # # print(y_r)
# # # print(x_u)
# # # print(x_d)
# # transformed_super = transformed_super[x_u: x_d, y_l: y_r]
#
# transformed_overlap = rigid_transform_tr(mask, rotation, translation, 'nearest')
# raw_mask = np.zeros((1024, 1024))
# raw_mask[300: 812, 300: 812] = 1
# transformed_overlap = transformed_overlap + raw_mask
# # cv2.imshow("raw + trans", transformed_overlap)
# transformed_overlap[transformed_overlap < 1.5] = 0
# transformed_overlap[transformed_overlap > 1.5] = 1
# deformed_mask = np.zeros(transformed_total.shape)
# im_merge = np.concatenate((transformed_total[..., None], deformed_mask[..., None]), axis=2)
# im_trans_mask = np.concatenate((transformed_overlap[..., None], transformed_overlap[..., None]), axis=2)
# im_t = None
# # indices = None
# count = 0
# while count < 3:
#     im_merge_t, indices = overlap_mask_elastic_transform(im_merge, im_trans_mask, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08)
#     im_t = im_merge_t[..., 0]
#     im_mask_t = im_merge_t[..., 1]
#     count += 1
#     im_merge = im_merge_t
#
# raw_right = im_t * mask
# transformed_right = rigid_transform_tr(raw_right, rotation, translation, 'bilinear', inv=True)
# mask_super = transformed_mask.copy()
# mask_super[300: 812, 300: 812] = 1
# transformed_super = transformed_right.copy()
# transformed_super[300: 812, 300: 812] = raw_left
# x_map = np.sum(mask_super, axis=1)
# y_map = np.sum(mask_super, axis=0)
# y_l = min(np.argwhere(y_map >= 1.0))[0]
# y_r = max(np.argwhere(y_map >= 1.0))[0]
# x_u = min(np.argwhere(x_map >= 1.0))[0]
# x_d = max(np.argwhere(x_map >= 1.0))[0]
# transformed_super = transformed_super[x_u: x_d, y_l: y_r]
# cv2.imshow("overlap_mask", transformed_overlap)
# cv2.imshow("elastic_total", im_t)
# cv2.imshow("total", total_img)
# cv2.imshow("trans_right", transformed_right)
# cv2.imshow("trans_mask", transformed_mask)
# cv2.imshow("mask_super", mask_super)
# cv2.imshow("transformed_super", transformed_super)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# path = "../data/sample_C_padded_20160501.mrc"
# with mrcfile.open(path) as mrc:
#     data = mrc.data
#
# print(data.shape)
# rest_index = []
# for i in range(len(data)):
#     if i+1 not in [52, 112, 124]:
#         rest_index.append(i)
# data = data[rest_index]
# mask_right_seq = []
# raw_seq = []
# transformed_seq = []
# for i in tqdm(range(len(rest_index))):
#     for patch_i in [0, 1024, 2048]:
#         for patch_j in [0, 1024, 2048]:
#             total_img = normalize_img(data[i, patch_i: patch_i + 1024, patch_j: patch_j + 1024])
#             raw = np.zeros((1024, 1024))
#             raw[300: 812, 300: 812] = total_img[300: 812, 300: 812]
#             raw_seq.append(raw)
#
#             rotation = random.uniform(5.0, 10.0) * random.choice([-1, 1])
#             translation = [random.uniform(50.0, 100.0) * random.choice([-1, 1]), random.uniform(50.0, 100.0) * random.choice([-1, 1])]
#
#             transformed_total = rigid_transform_tr(total_img, rotation, translation)
#             transformed_total = normalize_img(transformed_total)
#             # raw_right = np.zeros((1024, 1024))
#             # raw_right[300: 812, 300: 812] = transformed_total[300: 812, 300: 812]
#
#             mask = np.zeros((1024, 1024))
#             mask[300: 812, 300: 812] = 1
#             transformed_mask = rigid_transform_tr(mask, rotation, translation, 'nearest', inv=True)
#             mask_right_seq.append(transformed_mask)
#
#             transformed_overlap = rigid_transform_tr(mask, rotation, translation, 'nearest')
#             raw_mask = np.zeros((1024, 1024))
#             raw_mask[300: 812, 300: 812] = 1
#             transformed_overlap = transformed_overlap + raw_mask
#             # cv2.imshow("raw + trans", transformed_overlap)
#             transformed_overlap[transformed_overlap < 1.5] = 0
#             transformed_overlap[transformed_overlap > 1.5] = 1
#             deformed_mask = np.zeros(transformed_total.shape)
#             im_merge = np.concatenate((transformed_total[..., None], deformed_mask[..., None]), axis=2)
#             im_trans_mask = np.concatenate((transformed_overlap[..., None], transformed_overlap[..., None]), axis=2)
#             im_t = None
#             # indices = None
#             count = 0
#             while count < 4:
#                 im_merge_t, indices = overlap_mask_elastic_transform(im_merge, im_trans_mask, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08)
#                 im_t = im_merge_t[..., 0]
#                 im_mask_t = im_merge_t[..., 1]
#                 count += 1
#                 im_merge = im_merge_t
#             transformed_right = im_t * mask
#             transformed_right = rigid_transform_tr(transformed_right, rotation, translation, 'bilinear', inv=True)
#             transformed_seq.append(transformed_right)
#
# raw_data = np.stack(raw_seq, axis=0).astype(np.float32)
# transformed_data = np.stack(transformed_seq, axis=0).astype(np.float32)
# transformed_mask_data = np.stack(mask_right_seq, axis=0).astype(np.float32)
# with mrcfile.new("../data/train/raw_left.mrc", overwrite=True) as mrc:
#     mrc.set_data(raw_data)
# with mrcfile.new("../data/train/deformed_right.mrc", overwrite=True) as mrc:
#     mrc.set_data(transformed_data)
# with mrcfile.new("../data/train/mask_right.mrc", overwrite=True) as mrc:
#     mrc.set_data(transformed_mask_data)
#
# with open("../data/train/readme", 'w') as f:
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("name", "data_size", "real_content_size", "description"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("raw_left", str(raw_data.shape), "512*512", "one image"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("deformed_right", str(transformed_data.shape), "512*512", "deformed image"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("mask_right", str(transformed_mask_data.shape), "512*512", "position mask of raw_right"))


"""
slight train
"""
path = "../data/sample_C_padded_20160501.mrc"
with mrcfile.open(path) as mrc:
    data = mrc.data

print(data.shape)
rest_index = []
for i in range(len(data)):
    if i+1 not in [52, 112, 124]:
        rest_index.append(i)
data = data[rest_index]
mask_right_seq = []
raw_seq = []
transformed_seq = []
for i in tqdm(range(len(rest_index))):
    for patch_i in [0, 1024, 2048]:
        for patch_j in [0, 1024, 2048]:
            total_img = normalize_img(data[i, patch_i: patch_i + 1024, patch_j: patch_j + 1024])
            raw = np.zeros((1024, 1024))
            raw[300: 812, 300: 812] = total_img[300: 812, 300: 812]
            raw_seq.append(raw)

            rotation = random.uniform(5.0, 10.0) * random.choice([-1, 1])
            translation = [random.uniform(50.0, 100.0) * random.choice([-1, 1]), random.uniform(50.0, 100.0) * random.choice([-1, 1])]

            transformed_total = rigid_transform_tr(total_img, rotation, translation)
            transformed_total = normalize_img(transformed_total)
            # raw_right = np.zeros((1024, 1024))
            # raw_right[300: 812, 300: 812] = transformed_total[300: 812, 300: 812]

            mask = np.zeros((1024, 1024))
            mask[300: 812, 300: 812] = 1
            transformed_mask = rigid_transform_tr(mask, rotation, translation, 'nearest', inv=True)
            mask_right_seq.append(transformed_mask)

            transformed_overlap = rigid_transform_tr(mask, rotation, translation, 'nearest')
            raw_mask = np.zeros((1024, 1024))
            raw_mask[300: 812, 300: 812] = 1
            transformed_overlap = transformed_overlap + raw_mask
            # cv2.imshow("raw + trans", transformed_overlap)
            transformed_overlap[transformed_overlap < 1.5] = 0
            transformed_overlap[transformed_overlap > 1.5] = 1
            deformed_mask = np.zeros(transformed_total.shape)
            im_merge = np.concatenate((transformed_total[..., None], deformed_mask[..., None]), axis=2)
            im_trans_mask = np.concatenate((transformed_overlap[..., None], transformed_overlap[..., None]), axis=2)
            im_t = None
            # indices = None
            count = 0
            while count < 4:
                im_merge_t, indices = overlap_mask_elastic_transform(im_merge, im_trans_mask, im_merge.shape[1] * 0.2, im_merge.shape[1] * 0.04)
                im_t = im_merge_t[..., 0]
                im_mask_t = im_merge_t[..., 1]
                count += 1
                im_merge = im_merge_t
            transformed_right = im_t * mask
            transformed_right = rigid_transform_tr(transformed_right, rotation, translation, 'bilinear', inv=True)
            transformed_seq.append(transformed_right)

raw_data = np.stack(raw_seq, axis=0).astype(np.float32)
transformed_data = np.stack(transformed_seq, axis=0).astype(np.float32)
transformed_mask_data = np.stack(mask_right_seq, axis=0).astype(np.float32)
with mrcfile.new("../data/slight_train/raw_left.mrc", overwrite=True) as mrc:
    mrc.set_data(raw_data)
with mrcfile.new("../data/slight_train/deformed_right.mrc", overwrite=True) as mrc:
    mrc.set_data(transformed_data)
with mrcfile.new("../data/slight_train/mask_right.mrc", overwrite=True) as mrc:
    mrc.set_data(transformed_mask_data)

with open("../data/slight_train/readme", 'w') as f:
    f.write("{:<20} {:<20} {:<20} {:<20} \n".format("name", "data_size", "real_content_size", "description"))
    f.write("{:<20} {:<20} {:<20} {:<20} \n".format("raw_left", str(raw_data.shape), "512*512", "one image"))
    f.write("{:<20} {:<20} {:<20} {:<20} \n".format("deformed_right", str(transformed_data.shape), "512*512", "deformed image"))
    f.write("{:<20} {:<20} {:<20} {:<20} \n".format("mask_right", str(transformed_mask_data.shape), "512*512", "position mask of raw_right"))


"""
test
"""
# raw_path = "../data/train/raw_left.mrc"
# transformed_path = "../data/train/deformed_right.mrc"
# transformed__mask_path = "../data/train/mask_right.mrc"
# with mrcfile.open(raw_path) as mrc:
#     raw_data = mrc.data
# with mrcfile.open(transformed_path) as mrc:
#     transformed_data = mrc.data
# with mrcfile.open(transformed__mask_path) as mrc:
#     transformed_mask_data = mrc.data
# total_num = raw_data.shape[0]
# # index = 15
# # raw_img = raw_data[index]
# # transformed_img = transformed_data[index]
# # cv2.imshow("raw_img", raw_img)
# # cv2.imshow("transformed_img", transformed_img)
# # transformed_mask_img = transformed_mask_data[index]
# # raw_mask_img = np.zeros((1024, 1024))
# # raw_mask_img[300: 812, 300: 812] = 1
# # raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut = pre_process_cut(raw_img, transformed_img, raw_mask_img, transformed_mask_img)
# # cv2.imshow("raw_cut", raw_cut)
# # cv2.imshow("transformed_cut", transformed_cut)
# # cv2.imshow("raw_mask_cut", raw_mask_cut)
# # cv2.imshow("transformed_mask_cut", transformed_mask_cut)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()'
# raw_cut_list = []
# transformed_cut_list = []
# raw_mask_cut_list = []
# transformed_mask_cut_list = []
# for index in tqdm(range(total_num)):
#     raw_img = raw_data[index]
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     raw_mask_img = np.zeros((1024, 1024))
#     raw_mask_img[300: 812, 300: 812] = 1
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut = pre_process_cut(raw_img, transformed_img,
#                                                                                    raw_mask_img, transformed_mask_img)
#     raw_cut_list.append(raw_cut)
#     transformed_cut_list.append(transformed_cut)
#     raw_mask_cut_list.append(raw_mask_cut)
#     transformed_mask_cut_list.append(transformed_mask_cut)
# raw_cut_data = np.stack(raw_cut_list, axis=0)
# transformed_cut_data = np.stack(transformed_cut_list, axis=0)
# raw_mask_cut_data = np.stack(raw_mask_cut_list, axis=0)
# transformed_mask_cut_data = np.stack(transformed_mask_cut_list, axis=0)
# with mrcfile.new("../data/train/raw_left_cut.mrc", overwrite=True) as mrc:
#     mrc.set_data(raw_cut_data)
# with mrcfile.new("../data/train/deformed_right_cut.mrc", overwrite=True) as mrc:
#     mrc.set_data(transformed_cut_data)
# with mrcfile.new("../data/train/mask_left_cut.mrc", overwrite=True) as mrc:
#     mrc.set_data(raw_mask_cut_data)
# with mrcfile.new("../data/train/mask_right_cut.mrc", overwrite=True) as mrc:
#     mrc.set_data(transformed_mask_cut_data)


"""
test data generator
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
# seg_left = []
# seg_right = []
#
# mask_right_seq = []
# raw_seq = []
# transformed_seq = []
# transformed_mesh_seq = []
# for i in tqdm(range(len(rest_index))):
# # for i in [1, 2]:
#     total_img = normalize_img(data[i, :, :])
#     total_seg = seg[i, :, :]
#     raw = np.zeros((1024, 1024))
#     raw[300: 812, 300: 812] = total_img[300: 812, 300: 812]
#     raw_seq.append(raw)
#     raw_seg = np.zeros((1024, 1024))
#     raw_seg[300: 812, 300: 812] = total_seg[300: 812, 300: 812]
#     seg_left.append(raw_seg)
#
#     # rotation = random.uniform(5.0, 10.0) * random.choice([-1, 1])
#     rotation = 0.0
#     translation = [random.uniform(50.0, 100.0) * random.choice([-1, 1]), random.uniform(50.0, 100.0) * random.choice([-1, 1])]
#
#     transformed_total = rigid_transform_tr(total_img, rotation, translation)
#     transformed_total = normalize_img(transformed_total)
#     transformed_total_seg = rigid_transform_tr(total_seg, rotation, translation)
#
#     mask = np.zeros((1024, 1024))
#     mask[300: 812, 300: 812] = 1
#     mesh_grid_right = np.zeros((1024, 1024))
#     for i in range(300, 810, 50):
#         mesh_grid_right[i:i + 5, 300: 812] = 1.0
#         mesh_grid_right[300: 812, i:i + 5] = 1.0
#     transformed_mask = rigid_transform_tr(mask, rotation, translation, 'nearest', inv=True)
#     transformed_mesh_grid = rigid_transform_tr(mesh_grid_right, rotation, translation, 'nearest', inv=True)
#     mask_right_seq.append(transformed_mask)
#     transformed_mesh_seq.append(transformed_mesh_grid)
#
#     transformed_overlap = rigid_transform_tr(mask, rotation, translation, 'nearest')
#     raw_mask = np.zeros((1024, 1024))
#     raw_mask[300: 812, 300: 812] = 1
#     transformed_overlap = transformed_overlap + raw_mask
#     # cv2.imshow("raw + trans", transformed_overlap)
#     transformed_overlap[transformed_overlap < 1.5] = 0
#     transformed_overlap[transformed_overlap > 1.5] = 1
#     # deformed_mask = np.zeros(transformed_total.shape)
#     deformed_mask = transformed_total_seg
#     im_merge = np.concatenate((transformed_total[..., None], deformed_mask[..., None]), axis=2)
#     im_trans_mask = np.concatenate((transformed_overlap[..., None], transformed_overlap[..., None]), axis=2)
#     im_t = None
#     im_seg_t = None
#     # indices = None
#     count = 0
#     while count < 4:
#         im_merge_t, indices = overlap_mask_elastic_transform(im_merge, im_trans_mask, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08)
#         im_t = im_merge_t[..., 0]
#         im_seg_t = im_merge_t[..., 1]
#         count += 1
#         im_merge = im_merge_t
#     transformed_right = im_t * mask
#     transformed_right = rigid_transform_tr(transformed_right, rotation, translation, 'bilinear', inv=True)
#     transformed_seq.append(transformed_right)
#     transformed_right_seg = im_seg_t * mask
#     transformed_right_seg = rigid_transform_tr(transformed_right_seg, rotation, translation, 'bilinear', inv=True)
#     seg_right.append(transformed_right_seg)
#
# raw_data = np.stack(raw_seq, axis=0).astype(np.float32)
# transformed_data = np.stack(transformed_seq, axis=0).astype(np.float32)
# transformed_mask_data = np.stack(mask_right_seq, axis=0).astype(np.float32)
# transformed_mesh_data = np.stack(transformed_mesh_seq, axis=0).astype(np.float32)
# seg_left_data = np.stack(seg_left, axis=0).astype(np.float32)
# seg_right_data = np.stack(seg_right, axis=0).astype(np.float32)
#
# store_dir = "test_no_rotate"
# with mrcfile.new("../data/%s/raw_left.mrc" % store_dir, overwrite=True) as mrc:
#     mrc.set_data(raw_data)
# with mrcfile.new("../data/%s/deformed_right.mrc" % store_dir, overwrite=True) as mrc:
#     mrc.set_data(transformed_data)
# with mrcfile.new("../data/%s/mask_right.mrc" % store_dir, overwrite=True) as mrc:
#     mrc.set_data(transformed_mask_data)
# with mrcfile.new("../data/%s/mesh_grid_right.mrc" % store_dir, overwrite=True) as mrc:
#     mrc.set_data(transformed_mesh_data)
# np.save("../data/%s/raw_left_seg.npy" % store_dir, seg_left_data)
# np.save("../data/%s/deformed_right_seg.npy" % store_dir, seg_right_data)
#
#
# with open("../data/%s/readme" % store_dir, 'w') as f:
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("name", "data_size", "real_content_size", "description"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("raw_left", str(raw_data.shape), "512*512", "one image"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("deformed_right", str(transformed_data.shape), "512*512", "deformed image"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("mask_right", str(transformed_mask_data.shape), "512*512", "position mask of raw_right"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("mask_right", str(transformed_mesh_data.shape), "512*512",
#                                                     "mesh grid of raw_right"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("seg_left", str(seg_left_data.shape), "512*512",
#                                                     "segmentation of raw_left"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("seg_right", str(seg_right_data.shape), "512*512",
#                                                     "segmentation of raw_right"))


"""
different deformation generator
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
# random_list = [np.random.randint(100, 8000) for _ in range(len(rest_index))]
#
# for k in range(1, 7):
#     print("%d start -----------------------" % k)
#     seg_left = []
#     seg_right = []
#
#     mask_right_seq = []
#     raw_seq = []
#     transformed_seq = []
#     transformed_mesh_seq = []
#     for i in tqdm(range(len(rest_index))):
#     # for i in [1, 2]:
#         total_img = normalize_img(data[i, :, :])
#         total_seg = seg[i, :, :]
#         raw = np.zeros((1024, 1024))
#         raw[300: 812, 300: 812] = total_img[300: 812, 300: 812]
#         raw_seq.append(raw)
#         raw_seg = np.zeros((1024, 1024))
#         raw_seg[300: 812, 300: 812] = total_seg[300: 812, 300: 812]
#         seg_left.append(raw_seg)
#
#         rotation = random.uniform(5.0, 10.0) * random.choice([-1, 1])
#         # rotation = 0.0
#         translation = [random.uniform(50.0, 100.0) * random.choice([-1, 1]), random.uniform(50.0, 100.0) * random.choice([-1, 1])]
#
#         transformed_total = rigid_transform_tr(total_img, rotation, translation)
#         transformed_total = normalize_img(transformed_total)
#         transformed_total_seg = rigid_transform_tr(total_seg, rotation, translation)
#
#         mask = np.zeros((1024, 1024))
#         mask[300: 812, 300: 812] = 1
#         mesh_grid_right = np.zeros((1024, 1024))
#         for p in range(300, 810, 50):
#             mesh_grid_right[p:p + 5, 300: 812] = 1.0
#             mesh_grid_right[300: 812, p:p + 5] = 1.0
#         transformed_mask = rigid_transform_tr(mask, rotation, translation, 'nearest', inv=True)
#         transformed_mesh_grid = rigid_transform_tr(mesh_grid_right, rotation, translation, 'nearest', inv=True)
#         mask_right_seq.append(transformed_mask)
#         transformed_mesh_seq.append(transformed_mesh_grid)
#
#         transformed_overlap = rigid_transform_tr(mask, rotation, translation, 'nearest')
#         raw_mask = np.zeros((1024, 1024))
#         raw_mask[300: 812, 300: 812] = 1
#         transformed_overlap = transformed_overlap + raw_mask
#         # cv2.imshow("raw + trans", transformed_overlap)
#         transformed_overlap[transformed_overlap < 1.5] = 0
#         transformed_overlap[transformed_overlap > 1.5] = 1
#         # deformed_mask = np.zeros(transformed_total.shape)
#         deformed_mask = transformed_total_seg
#         im_merge = np.concatenate((transformed_total[..., None], deformed_mask[..., None]), axis=2)
#         im_trans_mask = np.concatenate((transformed_overlap[..., None], transformed_overlap[..., None]), axis=2)
#         im_t = None
#         im_seg_t = None
#         # indices = None
#         count = 0
#         while count < k:
#             im_merge_t, indices = overlap_mask_elastic_transform(im_merge, im_trans_mask, im_merge.shape[1] * 2,
#                                                                  im_merge.shape[1] * 0.08, random_state=np.random.RandomState(random_list[i]))
#             im_t = im_merge_t[..., 0]
#             im_seg_t = im_merge_t[..., 1]
#             count += 1
#             im_merge = im_merge_t
#         transformed_right = im_t * mask
#         transformed_right = rigid_transform_tr(transformed_right, rotation, translation, 'bilinear', inv=True)
#         transformed_seq.append(transformed_right)
#         transformed_right_seg = im_seg_t * mask
#         transformed_right_seg = rigid_transform_tr(transformed_right_seg, rotation, translation, 'bilinear', inv=True)
#         seg_right.append(transformed_right_seg)
#
#     raw_data = np.stack(raw_seq, axis=0).astype(np.float32)
#     transformed_data = np.stack(transformed_seq, axis=0).astype(np.float32)
#     transformed_mask_data = np.stack(mask_right_seq, axis=0).astype(np.float32)
#     transformed_mesh_data = np.stack(transformed_mesh_seq, axis=0).astype(np.float32)
#     seg_left_data = np.stack(seg_left, axis=0).astype(np.float32)
#     seg_right_data = np.stack(seg_right, axis=0).astype(np.float32)
#
#     store_dir = "different_deformation/%d" % k
#     with mrcfile.new("../data/%s/raw_left.mrc" % store_dir, overwrite=True) as mrc:
#         mrc.set_data(raw_data)
#     with mrcfile.new("../data/%s/deformed_right.mrc" % store_dir, overwrite=True) as mrc:
#         mrc.set_data(transformed_data)
#     with mrcfile.new("../data/%s/mask_right.mrc" % store_dir, overwrite=True) as mrc:
#         mrc.set_data(transformed_mask_data)
#     with mrcfile.new("../data/%s/mesh_grid_right.mrc" % store_dir, overwrite=True) as mrc:
#         mrc.set_data(transformed_mesh_data)
#     np.save("../data/%s/raw_left_seg.npy" % store_dir, seg_left_data)
#     np.save("../data/%s/deformed_right_seg.npy" % store_dir, seg_right_data)
#
#
#     with open("../data/%s/readme" % store_dir, 'w') as f:
#         f.write("{:<20} {:<20} {:<20} {:<20} \n".format("name", "data_size", "real_content_size", "description"))
#         f.write("{:<20} {:<20} {:<20} {:<20} \n".format("raw_left", str(raw_data.shape), "512*512", "one image"))
#         f.write("{:<20} {:<20} {:<20} {:<20} \n".format("deformed_right", str(transformed_data.shape), "512*512", "deformed image"))
#         f.write("{:<20} {:<20} {:<20} {:<20} \n".format("mask_right", str(transformed_mask_data.shape), "512*512", "position mask of raw_right"))
#         f.write("{:<20} {:<20} {:<20} {:<20} \n".format("mask_right", str(transformed_mesh_data.shape), "512*512",
#                                                         "mesh grid of raw_right"))
#         f.write("{:<20} {:<20} {:<20} {:<20} \n".format("seg_left", str(seg_left_data.shape), "512*512",
#                                                         "segmentation of raw_left"))
#         f.write("{:<20} {:<20} {:<20} {:<20} \n".format("seg_right", str(seg_right_data.shape), "512*512",
#                                                         "segmentation of raw_right"))


"""
arbitrary test generator 
"""

# path = "../data/sample_C_padded_20160501.mrc"
# with mrcfile.open(path) as mrc:
#     data = mrc.data
#
# print(data.shape)
# rest_index = []
# for i in range(len(data)):
#     if i+1 not in [52, 112, 124]:
#         rest_index.append(i)
# data = data[rest_index]
# index = 10
#
# total_img = normalize_img(data[index])
# raw = np.zeros((3072, 3072))
# raw[500: 1524, 500: 2548] = total_img[500: 1524, 500: 2548]
# # total_img = normalize_img(data[index, 512: -512, 512: -512])
# # raw = np.zeros((2048, 2048))
# # raw[500: 1524, 500: 1524] = total_img[500: 1524, 500: 1524]
# # total_img = normalize_img(data[index, 512: 1536, 512: 1536])
# # raw = np.zeros((1024, 1024))
# # raw[300: 812, 300: 812] = total_img[300: 812, 300: 812]
#
# rotation = random.uniform(5.0, 10.0) * random.choice([-1, 1])
# translation = [random.uniform(50.0, 100.0) * random.choice([-1, 1]), random.uniform(50.0, 100.0) * random.choice([-1, 1])]
#
# transformed_total = rigid_transform_tr(total_img, rotation, translation)
# transformed_total = normalize_img(transformed_total)
#
# # mask = np.zeros((1024, 1024))
# # mask[300: 812, 300: 812] = 1
# # mask = np.zeros((2048, 2048))
# # mask[500: 1524, 500: 1524] = 1
# mask = np.zeros((3072, 3072))
# mask[500: 1524, 500: 2548] = 1
# transformed_mask = rigid_transform_tr(mask, rotation, translation, 'nearest', inv=True)
#
# transformed_overlap = rigid_transform_tr(mask, rotation, translation, 'nearest')
# # raw_mask = np.zeros((1024, 1024))
# # raw_mask[300: 812, 300: 812] = 1
# # raw_mask = np.zeros((2048, 2048))
# # raw_mask[500: 1524, 500: 1524] = 1
# raw_mask = np.zeros((3072, 3072))
# raw_mask[500: 1524, 500: 2548] = 1
# transformed_overlap = transformed_overlap + raw_mask
# # cv2.imshow("raw + trans", transformed_overlap)
# transformed_overlap[transformed_overlap < 1.5] = 0
# transformed_overlap[transformed_overlap > 1.5] = 1
# deformed_mask = np.zeros(transformed_total.shape)
# im_merge = np.concatenate((transformed_total[..., None], deformed_mask[..., None]), axis=2)
# im_trans_mask = np.concatenate((transformed_overlap[..., None], transformed_overlap[..., None]), axis=2)
# im_t = None
# # indices = None
# count = 0
# while count < 4:
#     im_merge_t, indices = overlap_mask_elastic_transform(im_merge, im_trans_mask, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08)
#     im_t = im_merge_t[..., 0]
#     im_mask_t = im_merge_t[..., 1]
#     count += 1
#     im_merge = im_merge_t
# transformed_right = im_t * mask
# transformed_right = rigid_transform_tr(transformed_right, rotation, translation, 'bilinear', inv=True)
#
# # raw = normalize_img(raw)
# # transformed_right = normalize_img(transformed_right)
# # transformed_mask = normalize_img(transformed_mask)
# cv2.imshow("raw", raw)
# cv2.imshow("transformed_right", transformed_right)
# cv2.imshow("transformed_mask", transformed_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
test img check
"""

# with mrcfile.open("../data/test/raw_left.mrc") as mrc:
#     raw_data = mrc.data
# with mrcfile.open("../data/test/deformed_right.mrc") as mrc:
#     transformed_data = mrc.data
# with mrcfile.open("../data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
# with mrcfile.open("../data/test/mesh_grid_right.mrc") as mrc:
#     transformed_mesh_data = mrc.data
#
# seg_left_data = np.load("../data/test/raw_left_seg.npy")
# seg_right_data = np.load("../data/test/deformed_right_seg.npy")
#
# # print(np.max(seg_left_data[0]))
#
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
#
# index = 22
#
# raw_img = raw_data[index]
# transformed_img = transformed_data[index]
# transformed_img_mask = transformed_mask_data[index]
# transformed_img_mesh = transformed_mesh_data[index]
# transformed_img_mask = np.where(transformed_img_mask > 0, 1.0, 0)
# transformed_img_mesh = np.where(transformed_img_mesh > 0, 1.0, 0)
# raw_mask_img = np.zeros(raw_img.shape)
# raw_mask_img[raw_img > 0.1] = 1
# mask_added = raw_mask_img + transformed_img_mask
# mask_super = np.where(mask_added > 0.5, 1.0, 0)
# mask_overlap = np.where(mask_added > 1.5, 1.0, 0)
# raw_pad = raw_img + transformed_img * (transformed_img_mask - mask_overlap)
# transformed_pad = transformed_img + raw_img * (raw_mask_img - mask_overlap)
#
# gt = data[index] * mask_super
# seg_left_img = seg_left_data[index]
# seg_right_img = seg_right_data[index]
#
# # cv2.imshow("gt", np.uint8(gt))
# gt = normalize_img(gt)
# cv2.imshow("gt", gt)
#
# cv2.imshow("raw_pad", raw_pad)
# cv2.imshow("transformed_pad", transformed_pad)
# cv2.imshow("raw_left", raw_img)
# cv2.imshow("deformed_right", transformed_img)
# cv2.imshow("mask", transformed_img_mask)
# cv2.imshow("mesh_grid", transformed_img_mesh)
# # cv2.imshow("seg_left", seg_left_img)
# # cv2.imshow("seg_right", seg_right_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
real data train
"""

# path = "../data/new_train/sample.mrc"
# with mrcfile.open(path) as mrc:
#     data = mrc.data
#
# print(data.shape)
# train_num = data.shape[0]
# mask_right_seq = []
# raw_seq = []
# transformed_seq = []
# for i in tqdm(range(train_num)):
#     for patch_i in [0, 1024, 2048]:
#         for patch_j in [0, 1024, 2048]:
#             total_img = normalize_img(data[i, patch_i: patch_i + 1024, patch_j: patch_j + 1024])
#             raw = np.zeros((1024, 1024))
#             raw[300: 812, 300: 812] = total_img[300: 812, 300: 812]
#             raw_seq.append(raw)
#
#             rotation = random.uniform(0.0, 3.0) * (-1)
#             translation = [random.uniform(80.0, 150.0) * (-1), random.uniform(80.0, 150.0)]
#
#             transformed_total = rigid_transform_tr(total_img, rotation, translation)
#             transformed_total = normalize_img(transformed_total)
#             # raw_right = np.zeros((1024, 1024))
#             # raw_right[300: 812, 300: 812] = transformed_total[300: 812, 300: 812]
#
#             mask = np.zeros((1024, 1024))
#             mask[300: 812, 300: 812] = 1
#             transformed_mask = rigid_transform_tr(mask, rotation, translation, 'nearest', inv=True)
#             mask_right_seq.append(transformed_mask)
#
#             transformed_overlap = rigid_transform_tr(mask, rotation, translation, 'nearest')
#             raw_mask = np.zeros((1024, 1024))
#             raw_mask[300: 812, 300: 812] = 1
#             transformed_overlap = transformed_overlap + raw_mask
#             # cv2.imshow("raw + trans", transformed_overlap)
#             transformed_overlap[transformed_overlap < 1.5] = 0
#             transformed_overlap[transformed_overlap > 1.5] = 1
#             deformed_mask = np.zeros(transformed_total.shape)
#             im_merge = np.concatenate((transformed_total[..., None], deformed_mask[..., None]), axis=2)
#             im_trans_mask = np.concatenate((transformed_overlap[..., None], transformed_overlap[..., None]), axis=2)
#             im_t = None
#             # indices = None
#             count = 0
#             while count < 4:
#                 im_merge_t, indices = overlap_mask_elastic_transform(im_merge, im_trans_mask, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08)
#                 im_t = im_merge_t[..., 0]
#                 im_mask_t = im_merge_t[..., 1]
#                 count += 1
#                 im_merge = im_merge_t
#             transformed_right = im_t * mask
#             transformed_right = rigid_transform_tr(transformed_right, rotation, translation, 'bilinear', inv=True)
#             transformed_seq.append(transformed_right)
#
# raw_data = np.stack(raw_seq, axis=0).astype(np.float32)
# transformed_data = np.stack(transformed_seq, axis=0).astype(np.float32)
# transformed_mask_data = np.stack(mask_right_seq, axis=0).astype(np.float32)
# with mrcfile.new("../data/new_train/raw_left.mrc", overwrite=True) as mrc:
#     mrc.set_data(raw_data)
# with mrcfile.new("../data/new_train/deformed_right.mrc", overwrite=True) as mrc:
#     mrc.set_data(transformed_data)
# with mrcfile.new("../data/new_train/mask_right.mrc", overwrite=True) as mrc:
#     mrc.set_data(transformed_mask_data)
#
# with open("../data/new_train/readme", 'w') as f:
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("name", "data_size", "real_content_size", "description"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("raw_left", str(raw_data.shape), "512*512", "one image"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("deformed_right", str(transformed_data.shape), "512*512", "deformed image"))
#     f.write("{:<20} {:<20} {:<20} {:<20} \n".format("mask_right", str(transformed_mask_data.shape), "512*512", "position mask of raw_right"))

"""
real image pair
"""
# # path = "../data/sample_C_padded_20160501.mrc"
# path = "../data/test/real_3/total.png"
# data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#
# total_img = normalize_img(data)
# raw = np.zeros((1024, 1024))
# raw[300: 812, 300: 812] = total_img[300: 812, 300: 812]
#
# # rotation = random.uniform(0.0, 1.0) * (-1)
# rotation = 0
# # translation = [random.uniform(100.0, 150.0) * (-1), random.uniform(10.0, 50.0)]
# translation = [random.uniform(100.0, 150.0) * (-1), 0.0]
#
# transformed_total = rigid_transform_tr(total_img, rotation, translation)
# transformed_total = normalize_img(transformed_total)
#
# mask = np.zeros((1024, 1024))
# mask[300: 812, 300: 812] = 1
# transformed_mask = rigid_transform_tr(mask, rotation, translation, 'nearest', inv=True)
#
# transformed_overlap = rigid_transform_tr(mask, rotation, translation, 'nearest')
# raw_mask = np.zeros((1024, 1024))
# raw_mask[300: 812, 300: 812] = 1
# transformed_overlap = transformed_overlap + raw_mask
# # cv2.imshow("raw + trans", transformed_overlap)
# transformed_overlap[transformed_overlap < 1.5] = 0
# transformed_overlap[transformed_overlap > 1.5] = 1
# deformed_mask = np.zeros(transformed_total.shape)
# im_merge = np.concatenate((transformed_total[..., None], deformed_mask[..., None]), axis=2)
# im_trans_mask = np.concatenate((transformed_overlap[..., None], transformed_overlap[..., None]), axis=2)
# im_t = None
# im_seg_t = None
# # indices = None
# count = 0
# while count < 4:
#     im_merge_t, indices = overlap_mask_elastic_transform(im_merge, im_trans_mask, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08)
#     im_t = im_merge_t[..., 0]
#     im_seg_t = im_merge_t[..., 1]
#     count += 1
#     im_merge = im_merge_t
# transformed_right = im_t * mask
# transformed_right = rigid_transform_tr(transformed_right, rotation, translation, 'bilinear', inv=True)
#
# cv2.imshow("raw_left", raw)
# cv2.imshow("mask_left", mask)
# cv2.imshow("deformed_right", transformed_right)
# cv2.imshow("mask_right", transformed_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
