import numpy as np
import cv2
import os
import torch
from networks import *
from layers import SpatialTransformer
import mrcfile
from collections import defaultdict
from Utils import standardize_numpy, jacobian_determinant, window_NCC, Dice, pre_process, normalize_img, \
    normalize_numpy, pre_process_nostitch, stitch_add_mask_half, cal_Hausdorff, psnr
import pandas as pd
from tqdm import tqdm
import pytorch_ssim
import time
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0")

"""
cal DICE metric
"""
# data_dir = "data"
# test_dir = 'test'
# level_num = 3
# model_dir = "models/Overlap_prealign_%d" % level_num
#
#
# model = DualOverlap(level_num)
# # model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 10)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("data/test/raw_left.mrc") as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
# with mrcfile.open("data/test/deformed_right.mrc") as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# Dice_score = 0.0
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.001] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
#         model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)
#
#     left_align = left_align_level[-1]
#     right_align = right_align_level[-1]
#     stn = VariableSpatialTransformer()
#
#     seg_slice_left = seg_left_data[index]
#     seg_slice_left = torch.Tensor(seg_slice_left).view(-1, 1, 1024, 1024)
#     seg_slice_right = seg_right_data[index]
#     seg_slice_right = torch.Tensor(seg_slice_right).view(-1, 1, 1024, 1024)
#     moved_seg_right = stn(seg_slice_right, left_align, cpu_flag)
#     moved_seg_left = stn(seg_slice_left, right_align, cpu_flag)
#     moved_seg_left = moved_seg_left.cpu().data.numpy()
#     moved_seg_right = moved_seg_right.cpu().data.numpy()
#     moved_seg_left = np.squeeze(moved_seg_left)
#     moved_seg_right = np.squeeze(moved_seg_right)
#
#     moved_seg_left = np.where(moved_seg_left > 40, 255.0, 0.0)
#     moved_seg_right = np.where(moved_seg_right > 40, 255.0, 0.0)
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     # cv2.imshow("left_seg", (moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.imshow("right_seg", (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # print(Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r]))
#     Dice_score += Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#
# print(Dice_score / train_num)

"""
Dice noise
"""
# data_dir = "data"
# test_dir = 'test'
# level_num = 3
# model_dir = "models/Overlap_prealign_%d" % level_num
#
# model = DualOverlap(level_num)
# # model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 25)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("data/test/raw_left.mrc") as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
# with mrcfile.open("data/test/deformed_right.mrc") as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# Dice_score = 0.0
# HD_score = 0.0
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.001] = 1
#
#     raw_left_noised = raw_img + np.random.normal(0.0, 5.0 / 255.0, (1024, 1024))
#     raw_left_noised = np.where(raw_left_noised > 1.0, 1.0, raw_left_noised)
#     raw_left_noised = np.where(raw_left_noised < 0.0, 0.0, raw_left_noised)
#     raw_left_noised = raw_left_noised * raw_mask_img
#
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_left_noised,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
#         model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)
#
#     left_align = left_align_level[-1]
#     right_align = right_align_level[-1]
#     stn = VariableSpatialTransformer()
#
#     seg_slice_left = seg_left_data[index]
#     seg_slice_left = torch.Tensor(seg_slice_left).view(-1, 1, 1024, 1024)
#     seg_slice_right = seg_right_data[index]
#     seg_slice_right = torch.Tensor(seg_slice_right).view(-1, 1, 1024, 1024)
#     moved_seg_right = stn(seg_slice_right, left_align, cpu_flag)
#     moved_seg_left = stn(seg_slice_left, right_align, cpu_flag)
#     moved_seg_left = moved_seg_left.cpu().data.numpy()
#     moved_seg_right = moved_seg_right.cpu().data.numpy()
#     moved_seg_left = np.squeeze(moved_seg_left)
#     moved_seg_right = np.squeeze(moved_seg_right)
#
#     moved_seg_left = np.where(moved_seg_left > 40, 255.0, 0.0)
#     moved_seg_right = np.where(moved_seg_right > 40, 255.0, 0.0)
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     # cv2.imshow("left_seg", (moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.imshow("right_seg", (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # print(Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r]))
#     Dice_score += Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#     HD_score += cal_Hausdorff((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#
# print("Dice: ", Dice_score / train_num)
# print("Hausdorff: ", HD_score / train_num)


"""
Dice deformation
"""
# data_dir = "data"
# test_dir = 'different_deformation'
# degree = 6
# level_num = 3
# model_dir = "models/Overlap_prealign_%d" % level_num
#
# model = DualOverlap(level_num)
# # model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 25)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("%s/%s/%d/raw_left.mrc" % (data_dir, test_dir, degree)) as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
# with mrcfile.open("%s/%s/%d/deformed_right.mrc" % (data_dir, test_dir, degree)) as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("%s/%s/%d/mask_right.mrc" % (data_dir, test_dir, degree)) as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("%s/%s/%d/raw_left_seg.npy" % (data_dir, test_dir, degree))
# seg_right_data = np.load("%s/%s/%d/deformed_right_seg.npy" % (data_dir, test_dir, degree))
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# Dice_score = 0.0
# HD_score = 0.0
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.001] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
#         model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)
#
#     left_align = left_align_level[-1]
#     right_align = right_align_level[-1]
#     stn = VariableSpatialTransformer()
#
#     seg_slice_left = seg_left_data[index]
#     seg_slice_left = torch.Tensor(seg_slice_left).view(-1, 1, 1024, 1024)
#     seg_slice_right = seg_right_data[index]
#     seg_slice_right = torch.Tensor(seg_slice_right).view(-1, 1, 1024, 1024)
#     moved_seg_right = stn(seg_slice_right, left_align, cpu_flag)
#     moved_seg_left = stn(seg_slice_left, right_align, cpu_flag)
#     moved_seg_left = moved_seg_left.cpu().data.numpy()
#     moved_seg_right = moved_seg_right.cpu().data.numpy()
#     moved_seg_left = np.squeeze(moved_seg_left)
#     moved_seg_right = np.squeeze(moved_seg_right)
#
#     moved_seg_left = np.where(moved_seg_left > 40, 255.0, 0.0)
#     moved_seg_right = np.where(moved_seg_right > 40, 255.0, 0.0)
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     # cv2.imshow("left_seg", (moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.imshow("right_seg", (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # print(Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r]))
#     Dice_score += Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#     HD_score += cal_Hausdorff((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#
# print("Dice: ", Dice_score / train_num)
# print("Hausdorff: ", HD_score / train_num)


"""
DICE_unidirection
"""
# data_dir = "data"
# test_dir = 'test'
# level_num = 3
# model_dir = "models/Overlap_prealign_uni_%d" % level_num
#
#
# model = UniOverlap(level_num)
# # model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 25)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("data/test/raw_left.mrc") as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
# with mrcfile.open("data/test/deformed_right.mrc") as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# Dice_score = 0.0
# HD_score = 0.0
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.1] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     left_align_level, left_pad_mask_level = model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag)
#
#     left_align = left_align_level[-1]
#     stn = VariableSpatialTransformer()
#
#     seg_slice_left = seg_left_data[index]
#     seg_slice_left = torch.Tensor(seg_slice_left).view(-1, 1, 1024, 1024)
#     seg_slice_right = seg_right_data[index]
#     seg_slice_right = torch.Tensor(seg_slice_right).view(-1, 1, 1024, 1024)
#     moved_seg_right = stn(seg_slice_right, left_align, cpu_flag)
#     moved_seg_left = seg_slice_left
#     moved_seg_left = moved_seg_left.cpu().data.numpy()
#     moved_seg_right = moved_seg_right.cpu().data.numpy()
#     moved_seg_left = np.squeeze(moved_seg_left)
#     moved_seg_right = np.squeeze(moved_seg_right)
#
#     moved_seg_left = np.where(moved_seg_left > 40, 255.0, 0.0)
#     moved_seg_right = np.where(moved_seg_right > 40, 255.0, 0.0)
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     # cv2.imshow("left_seg", (moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.imshow("right_seg", (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # print(Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r]))
#     Dice_score += Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#
#     HD_score += cal_Hausdorff((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#
# print("Dice: ", Dice_score / train_num)
# print("Hausdorff: ", HD_score / train_num)


"""
Dice context exchange
"""
# data_dir = "data"
# test_dir = 'test'
# level_num = 3
# model_dir = "models/Overlap_prealign_nostitch_%d" % level_num
#
#
# model = DualOverlap(level_num)
# # model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 25)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("data/test/raw_left.mrc") as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
# with mrcfile.open("data/test/deformed_right.mrc") as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# Dice_score = 0.0
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.001] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process_nostitch(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
#         model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)
#
#     left_align = left_align_level[-1]
#     right_align = right_align_level[-1]
#     stn = VariableSpatialTransformer()
#
#     seg_slice_left = seg_left_data[index]
#     seg_slice_left = torch.Tensor(seg_slice_left).view(-1, 1, 1024, 1024)
#     seg_slice_right = seg_right_data[index]
#     seg_slice_right = torch.Tensor(seg_slice_right).view(-1, 1, 1024, 1024)
#     moved_seg_right = stn(seg_slice_right, left_align, cpu_flag)
#     moved_seg_left = stn(seg_slice_left, right_align, cpu_flag)
#     moved_seg_left = moved_seg_left.cpu().data.numpy()
#     moved_seg_right = moved_seg_right.cpu().data.numpy()
#     moved_seg_left = np.squeeze(moved_seg_left)
#     moved_seg_right = np.squeeze(moved_seg_right)
#
#     moved_seg_left = np.where(moved_seg_left > 40, 255.0, 0.0)
#     moved_seg_right = np.where(moved_seg_right > 80, 255.0, 0.0)
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     # cv2.imshow("left_seg", (moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.imshow("right_seg", (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # print(Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r]))
#     Dice_score += Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#
# print(Dice_score / train_num)

"""
Dice_Unet
"""

# data_dir = "data"
# test_dir = 'test'
# model_dir = "models"
# level_num = 3
#
# enc_nf = [16, 32, 32, 32]
# dec_nf = [32, 32, 32, 32, 32, 16, 16]
# model = VxmDense(
#         inshape=[1024, 1024],
#         nb_unet_features=[enc_nf, dec_nf],
#         bidir=False,
#         int_steps=7,
#         int_downsize=2)
#
# raw_path = "%s/%s/raw_left.mrc" % (data_dir, test_dir)
# deformed_path = "%s/%s/deformed_right.mrc" % (data_dir, test_dir)
# deformed_mask = "%s/%s/mask_right.mrc" % (data_dir, test_dir)
#
# epoch = 50
# model_dir = "%s/Overlap_prealign_unet_3/net_params_%d.pkl" % (model_dir, epoch)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open(raw_path) as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
#     raw_data = normalize_numpy(raw_data)
# with mrcfile.open(deformed_path) as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # deformed_data = standardize_numpy(transformed_data)
#     transformed_data = normalize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# Dice_score = 0.0
# HD_score = 0.0
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.1] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     _, deformation = model(transformed_cut, raw_cut, transformed_mask_cut)
#     stn = VariableSpatialTransformer()
#
#     seg_slice_left = seg_left_data[index]
#     seg_slice_left = torch.Tensor(seg_slice_left).view(-1, 1, 1024, 1024)
#     seg_slice_right = seg_right_data[index]
#     seg_slice_right = torch.Tensor(seg_slice_right).view(-1, 1, 1024, 1024)
#     moved_seg_right = stn(seg_slice_right, deformation, cpu_flag)
#     moved_seg_right = moved_seg_right.cpu().data.numpy()
#     moved_seg_right = np.squeeze(moved_seg_right)
#     moved_seg_left = seg_slice_left.cpu().data.numpy()
#     moved_seg_left = np.squeeze(moved_seg_left)
#
#     moved_seg_left = np.where(moved_seg_left > 40, 255.0, 0.0)
#     moved_seg_right = np.where(moved_seg_right > 40, 255.0, 0.0)
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     Dice_score += Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#     HD_score += cal_Hausdorff((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#
# print("Dice: ", Dice_score / train_num)
# print("Hausdorff: ", HD_score / train_num)


"""
Dice_Unet_no_stitch
"""

# data_dir = "data"
# test_dir = 'test'
# model_dir = "models"
# level_num = 3
#
# enc_nf = [16, 32, 32, 32]
# dec_nf = [32, 32, 32, 32, 32, 16, 16]
# model = VxmDense(
#         inshape=[1024, 1024],
#         nb_unet_features=[enc_nf, dec_nf],
#         bidir=False,
#         int_steps=7,
#         int_downsize=2)
#
# raw_path = "%s/%s/raw_left.mrc" % (data_dir, test_dir)
# deformed_path = "%s/%s/deformed_right.mrc" % (data_dir, test_dir)
# deformed_mask = "%s/%s/mask_right.mrc" % (data_dir, test_dir)
#
# epoch = 50
# model_dir = "%s/Overlap_prealign_unet_3/net_params_%d.pkl" % (model_dir, epoch)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open(raw_path) as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
#     raw_data = normalize_numpy(raw_data)
# with mrcfile.open(deformed_path) as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # deformed_data = standardize_numpy(deformed_data)
#     transformed_data = normalize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# Dice_score = 0.0
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.1] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut = raw_img, transformed_img, raw_mask_img, transformed_mask_img
#     mask_added = raw_mask_img + transformed_mask_img
#     mask_super = np.where(mask_added > 0.5, 1.0, 0)
#     mask_overlap = np.where(mask_added > 1.5, 1.0, 0)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#
#     cpu_flag = True
#     model.eval()
#     _, deformation = model(transformed_cut, raw_cut, transformed_mask_cut)
#     stn = VariableSpatialTransformer()
#
#     seg_slice_left = seg_left_data[index]
#     seg_slice_left = torch.Tensor(seg_slice_left).view(-1, 1, 1024, 1024)
#     seg_slice_right = seg_right_data[index]
#     seg_slice_right = torch.Tensor(seg_slice_right).view(-1, 1, 1024, 1024)
#     moved_seg_right = stn(seg_slice_right, deformation, cpu_flag)
#     moved_seg_right = moved_seg_right.cpu().data.numpy()
#     moved_seg_right = np.squeeze(moved_seg_right)
#     moved_seg_left = seg_slice_left.cpu().data.numpy()
#     moved_seg_left = np.squeeze(moved_seg_left)
#
#     moved_seg_left = np.where(moved_seg_left > 40, 255.0, 0.0)
#     moved_seg_right = np.where(moved_seg_right > 40, 255.0, 0.0)
#
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     # cv2.imshow("left_seg", (moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.imshow("right_seg", (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # print(Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r]))
#     Dice_score += Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#
# print(Dice_score / train_num)


"""
Cal NCC
"""

# #data_dir = "data"
# test_dir = 'test'
# level_num = 3
# model_dir = "models/Overlap_prealign_%d" % level_num
#
# model = DualOverlap(level_num)
# # model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 10)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("data/test/raw_left.mrc") as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     raw_data = standardize_numpy(raw_data)
# with mrcfile.open("data/test/deformed_right.mrc") as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# NCC_score = 0.0
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.1] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
#         model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)
#
#     left_align = left_align_level[-1]
#     right_align = right_align_level[-1]
#     stn = VariableSpatialTransformer()
#
#     raw_img_cut = torch.Tensor(raw_img_cut).float()
#     transformed_img_cut = torch.Tensor(transformed_img_cut).float()
#     raw_img_cut = raw_img_cut.view((-1, 1, *size))
#     transformed_img_cut = transformed_img_cut.view((-1, 1, *size))
#
#     left_deformed_data = stn(transformed_img_cut * transformed_mask_cut, left_align, cpu_flag)
#     right_deformed_data = stn(raw_img_cut * raw_mask_cut, right_align, cpu_flag)
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     # cv2.imshow("left_seg", (left_deformed_data * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.imshow("right_seg", (right_deformed_data * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # print(window_NCC(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r], (1, 23, 23)).data)
#
#     NCC_score += window_NCC(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r], (1, 35, 35)).data
#
# print(NCC_score / train_num)


"""
NCC_unidirection
"""

# #data_dir = "data"
# test_dir = 'test'
# model_dir = "models/Overlap_prealign_uni_3"
# level_num = 3
#
# model = UniOverlap(level_num)
# # model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 30)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("data/test/raw_left.mrc") as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
# with mrcfile.open("data/test/deformed_right.mrc") as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# NCC_score = 0.0
# SSIM_score = 0.0
#
# ssim_loss = pytorch_ssim.SSIM(window_size=11)
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.01] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     left_align_level, left_pad_mask_level = model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag)
#
#     left_align = left_align_level[-1]
#     stn = VariableSpatialTransformer()
#
#     raw_img_cut = torch.Tensor(raw_img_cut).float()
#     transformed_img_cut = torch.Tensor(transformed_img_cut).float()
#     raw_img_cut = raw_img_cut.view((-1, 1, *size))
#     transformed_img_cut = transformed_img_cut.view((-1, 1, *size))
#
#     left_deformed_data = stn(transformed_img_cut * transformed_mask_cut, left_align, cpu_flag)
#     right_deformed_data = raw_img_cut * raw_mask_cut
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     NCC_score += window_NCC(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r], (1, 35, 35)).data
#     SSIM_score += ssim_loss(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r]).item()
#
# print("NCC: ", NCC_score / train_num)
# print("SSIM: ", SSIM_score / train_num)

"""
NCC deformation
"""
# data_dir = "data"
# test_dir = 'different_deformation'
# degree = 6
# level_num = 3
# model_dir = "models/Overlap_prealign_%d" % level_num
#
# model = DualOverlap(level_num)
# # model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 25)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("%s/%s/%d/raw_left.mrc" % (data_dir, test_dir, degree)) as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
# with mrcfile.open("%s/%s/%d/deformed_right.mrc" % (data_dir, test_dir, degree)) as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("%s/%s/%d/mask_right.mrc" % (data_dir, test_dir, degree)) as mrc:
#     transformed_mask_data = mrc.data
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# NCC_score = 0.0
# SSIM_score = 0.0
#
# ssim_loss = pytorch_ssim.SSIM(window_size=11)
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.1] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
#         model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)
#
#     left_align = left_align_level[-1]
#     right_align = right_align_level[-1]
#     stn = VariableSpatialTransformer()
#
#     raw_img_cut = torch.Tensor(raw_img_cut).float()
#     transformed_img_cut = torch.Tensor(transformed_img_cut).float()
#     raw_img_cut = raw_img_cut.view((-1, 1, *size))
#     transformed_img_cut = transformed_img_cut.view((-1, 1, *size))
#
#     left_deformed_data = stn(transformed_img_cut * transformed_mask_cut, left_align, cpu_flag)
#     right_deformed_data = stn(raw_img_cut * raw_mask_cut, right_align, cpu_flag)
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     NCC_score += window_NCC(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r], (1, 23, 23)).data
#     SSIM_score += ssim_loss(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r]).item()
#
# print("NCC: ", NCC_score / train_num)
# print("SSIM: ", SSIM_score / train_num)

"""
NCC nostitch
"""
# data_dir = "data"
# test_dir = 'test'
# model_dir = "models/Overlap_prealign_nostitch_3"
# level_num = 3
#
# model = DualOverlap(level_num)
# # model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 30)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("data/test/raw_left.mrc") as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
# with mrcfile.open("data/test/deformed_right.mrc") as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# NCC_score = 0.0
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.1] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process_nostitch(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
#         model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)
#
#     left_align = left_align_level[-1]
#     right_align = right_align_level[-1]
#     stn = VariableSpatialTransformer()
#
#     raw_img_cut = torch.Tensor(raw_img_cut).float()
#     transformed_img_cut = torch.Tensor(transformed_img_cut).float()
#     raw_img_cut = raw_img_cut.view((-1, 1, *size))
#     transformed_img_cut = transformed_img_cut.view((-1, 1, *size))
#
#     left_deformed_data = stn(transformed_img_cut * transformed_mask_cut, left_align, cpu_flag)
#     right_deformed_data = stn(raw_img_cut * raw_mask_cut, right_align, cpu_flag)
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     # cv2.imshow("left_seg", (left_deformed_data * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.imshow("right_seg", (right_deformed_data * mask_overlap)[x_u: x_d, y_l: y_r])
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # print(window_NCC(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r], (1, 23, 23)).data)
#
#     NCC_score += window_NCC(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r], (1, 23, 23)).data
#
# print(NCC_score / train_num)


"""
NCC_unet
"""

# data_dir = "data"
# test_dir = 'test'
# model_dir = "models"
# level_num = 3
#
# enc_nf = [16, 32, 32, 32]
# dec_nf = [32, 32, 32, 32, 32, 16, 16]
# model = VxmDense(
#         inshape=[1024, 1024],
#         nb_unet_features=[enc_nf, dec_nf],
#         bidir=False,
#         int_steps=7,
#         int_downsize=2)
#
# raw_path = "%s/%s/raw_left.mrc" % (data_dir, test_dir)
# deformed_path = "%s/%s/deformed_right.mrc" % (data_dir, test_dir)
# deformed_mask = "%s/%s/mask_right.mrc" % (data_dir, test_dir)
#
# epoch = 50
# model_dir = "%s/Overlap_prealign_unet_3/net_params_%d.pkl" % (model_dir, epoch)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open(raw_path) as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
#     raw_data = normalize_numpy(raw_data)
# with mrcfile.open(deformed_path) as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # deformed_data = standardize_numpy(transformed_data)
#     transformed_data = normalize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# NCC_score = 0.0
# SSIM_score = 0.0
#
# ssim_loss = pytorch_ssim.SSIM(window_size=11)
#
# for index in tqdm(range(train_num)):
# # for index in [0]:
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.1] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     _, deformation = model(transformed_cut, raw_cut, transformed_mask_cut)
#     stn = VariableSpatialTransformer()
#
#     left_deformed_data = stn(transformed_cut * transformed_mask_cut, deformation * transformed_mask_cut, cpu_flag)
#     right_deformed_data = raw_cut * raw_mask_cut
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     NCC_score += window_NCC(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r], (1, 35, 35)).data
#     SSIM_score += ssim_loss(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r]).item()
#
# print("NCC: ", NCC_score / train_num)
# print("SSIM: ", SSIM_score / train_num)


"""
iterate cal
"""

# data_dir = "data"
# test_dir = 'test'
# level_num = 3
# model_dir = "models/Overlap_prealign_NCC_%d" % level_num
#
# model = DualOverlap(level_num)
# # model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 25)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("data/test/raw_left.mrc") as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
# with mrcfile.open("data/test/deformed_right.mrc") as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# new_left = []
# new_right = []
# seg_left = []
# seg_right = []
# mask_left_data = []
# mask_right_data = []
#
# stn = VariableSpatialTransformer()
# m_stn = VariableSpatialTransformer(mode='nearest')
# for index in tqdm(range(train_num)):
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.1] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
#         model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)
#
#     left_align = left_align_level[-1]
#     right_align = right_align_level[-1]
#
#     raw_img_cut = torch.Tensor(raw_img_cut).float()
#     transformed_img_cut = torch.Tensor(transformed_img_cut).float()
#     raw_img_cut = raw_img_cut.view((-1, 1, *size))
#     transformed_img_cut = transformed_img_cut.view((-1, 1, *size))
#
#     left_deformed_data = stn(transformed_img_cut * transformed_mask_cut, left_align, cpu_flag)
#     right_deformed_data = stn(raw_img_cut * raw_mask_cut, right_align, cpu_flag)
#     left_deformed_mask = m_stn(transformed_mask_cut, left_align, cpu_flag)
#     right_deformed_mask = m_stn(raw_mask_cut, right_align, cpu_flag)
#
#     seg_slice_left = seg_left_data[index]
#     seg_slice_left = torch.Tensor(seg_slice_left).view(-1, 1, 1024, 1024)
#     seg_slice_right = seg_right_data[index]
#     seg_slice_right = torch.Tensor(seg_slice_right).view(-1, 1, 1024, 1024)
#     moved_seg_right = stn(seg_slice_right, left_align, cpu_flag)
#     moved_seg_left = stn(seg_slice_left, right_align, cpu_flag)
#     moved_seg_left = moved_seg_left.cpu().data.numpy()
#     moved_seg_right = moved_seg_right.cpu().data.numpy()
#     moved_seg_left = np.squeeze(moved_seg_left)
#     moved_seg_right = np.squeeze(moved_seg_right)
#
#     # moved_seg_left = np.where(moved_seg_left > 40, 255.0, 0.0)
#     # moved_seg_right = np.where(moved_seg_right > 40, 255.0, 0.0)
#     # mask_super = mask_super.cpu().data.numpy()
#     # mask_super = np.squeeze(mask_super)
#     # mask_overlap = mask_overlap.cpu().data.numpy()
#     # mask_overlap = np.squeeze(mask_overlap)
#     # x_map = np.sum(mask_overlap, axis=1)
#     # y_map = np.sum(mask_overlap, axis=0)
#     # y_l = min(np.argwhere(y_map >= 1.0))[0]
#     # y_r = max(np.argwhere(y_map >= 1.0))[0]
#     # x_u = min(np.argwhere(x_map >= 1.0))[0]
#     # x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     # Dice_score += Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r],
#     #                    (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#     # NCC_score += window_NCC(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r],
#     #                         (1, 23, 23)).data
#
#     left_deformed_data = left_deformed_data.cpu().data.numpy()
#     left_deformed_data = np.squeeze(left_deformed_data)
#     right_deformed_data = right_deformed_data.cpu().data.numpy()
#     right_deformed_data = np.squeeze(right_deformed_data)
#     left_deformed_mask = left_deformed_mask.cpu().data.numpy()
#     left_deformed_mask = np.squeeze(left_deformed_mask)
#     right_deformed_mask = right_deformed_mask.cpu().data.numpy()
#     right_deformed_mask = np.squeeze(right_deformed_mask)
#     # left_deformed_data = normalize_img(left_deformed_data)
#     # right_deformed_data = normalize_img(right_deformed_data)
#
#     new_left.append(left_deformed_data)
#     new_right.append(right_deformed_data)
#     seg_left.append(moved_seg_left)
#     seg_right.append(moved_seg_right)
#     mask_left_data.append(left_deformed_mask)
#     mask_right_data.append(right_deformed_mask)
#
# np.save("%s/%s/iteration/%d/deformed_left.npy" % (data_dir, test_dir, 1), new_left)
# np.save("%s/%s/iteration/%d/deformed_right.npy" % (data_dir, test_dir, 1), new_right)
# np.save("%s/%s/iteration/%d/seg_left.npy" % (data_dir, test_dir, 1), seg_left)
# np.save("%s/%s/iteration/%d/seg_right.npy" % (data_dir, test_dir, 1), seg_right)
# np.save("%s/%s/iteration/%d/mask_left.npy" % (data_dir, test_dir, 1), mask_left_data)
# np.save("%s/%s/iteration/%d/mask_right.npy" % (data_dir, test_dir, 1), mask_right_data)


data_dir = "data"
test_dir = 'test'
level_num = 3
model_dir = "models/Overlap_prealign_slight_%d" % level_num

model = DualOverlap(level_num)
# model = model.to(device)

model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 25)
model.load_state_dict(torch.load(model_dir))

raw_data = np.load("%s/%s/iteration/%d/deformed_left.npy" % (data_dir, test_dir, 1))
deformed_data = np.load("%s/%s/iteration/%d/deformed_right.npy" % (data_dir, test_dir, 1))
seg_left = np.load("%s/%s/iteration/%d/seg_left.npy" % (data_dir, test_dir, 1))
seg_right = np.load("%s/%s/iteration/%d/seg_right.npy" % (data_dir, test_dir, 1))
mask_left_data = np.load("%s/%s/iteration/%d/mask_left.npy" % (data_dir, test_dir, 1))
mask_right_data = np.load("%s/%s/iteration/%d/mask_right.npy" % (data_dir, test_dir, 1))

train_num = raw_data.shape[0]
img_size = raw_data.shape[1:]
new_left = []
new_right = []

temp_raw_data = [None for _ in range(train_num)]
temp_deformed_data = [None for _ in range(train_num)]
temp_seg_left = [None for _ in range(train_num)]
temp_seg_right = [None for _ in range(train_num)]
temp_mask_left_data = [None for _ in range(train_num)]
temp_mask_right_data = [None for _ in range(train_num)]

stn = VariableSpatialTransformer()
m_stn = VariableSpatialTransformer(mode='nearest')
ssim_loss = pytorch_ssim.SSIM(window_size=11)

for iterate_num in range(2, 9):
    Dice_score = 0.0
    NCC_score = 0.0
    HD_score = 0.0
    SSIM_score = 0.0
    for index in tqdm(range(train_num)):

        final_left = raw_data[index] if iterate_num == 2 else temp_raw_data[index]
        final_right = deformed_data[index] if iterate_num == 2 else temp_deformed_data[index]
        final_left_seg = seg_left[index] if iterate_num == 2 else temp_seg_left[index]
        final_right_seg = seg_right[index] if iterate_num == 2 else temp_seg_right[index]
        final_mask_left = mask_left_data[index] if iterate_num == 2 else temp_mask_left_data[index]
        final_mask_right = mask_right_data[index] if iterate_num == 2 else temp_mask_right_data[index]

        final_left = standardize_numpy(final_left)
        final_right = standardize_numpy(final_right)

        final_mask_left = np.where(final_mask_left > 0, 1.0, 0)
        final_mask_right = np.where(final_mask_right > 0, 1.0, 0)

        raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(final_left,
                                                                                                             final_right,
                                                                                                             final_mask_left,
                                                                                                             final_mask_right)

        raw_img_cut = raw_cut.copy()
        transformed_img_cut = transformed_cut.copy()
        raw_cut = standardize_numpy(raw_cut)
        transformed_cut = standardize_numpy(transformed_cut)
        size = raw_cut.shape
        raw_cut = torch.Tensor(raw_cut).float()
        transformed_cut = torch.Tensor(transformed_cut).float()
        raw_mask_cut = torch.Tensor(raw_mask_cut).float()
        transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
        mask_super = torch.Tensor(mask_super).float()
        mask_overlap = torch.Tensor(mask_overlap).float()

        raw_cut = raw_cut.view((-1, 1, *size))
        transformed_cut = transformed_cut.view((-1, 1, *size))
        raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
        transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
        mask_super = mask_super.view(-1, 1, *size)
        mask_overlap = mask_overlap.view(-1, 1, *size)

        cpu_flag = True
        model.eval()
        left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
            model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)

        left_align = left_align_level[-1]
        right_align = right_align_level[-1]

        raw_img_cut = torch.Tensor(raw_img_cut).float()
        transformed_img_cut = torch.Tensor(transformed_img_cut).float()
        raw_img_cut = raw_img_cut.view((-1, 1, *size))
        transformed_img_cut = transformed_img_cut.view((-1, 1, *size))

        left_deformed_data = stn(transformed_img_cut * transformed_mask_cut, left_align, cpu_flag)
        right_deformed_data = stn(raw_img_cut * raw_mask_cut, right_align, cpu_flag)
        left_deformed_mask = m_stn(transformed_mask_cut, left_align, cpu_flag)
        right_deformed_mask = m_stn(raw_mask_cut, right_align, cpu_flag)

        seg_slice_left = torch.Tensor(final_left_seg).view(-1, 1, 1024, 1024)
        seg_slice_right = torch.Tensor(final_right_seg).view(-1, 1, 1024, 1024)
        moved_seg_right = stn(seg_slice_right, left_align, cpu_flag)
        moved_seg_left = stn(seg_slice_left, right_align, cpu_flag)
        moved_seg_left = moved_seg_left.cpu().data.numpy()
        moved_seg_right = moved_seg_right.cpu().data.numpy()
        moved_seg_left = np.squeeze(moved_seg_left)
        moved_seg_right = np.squeeze(moved_seg_right)

        moved_seg_left = np.where(moved_seg_left > 40, 255.0, 0.0)
        moved_seg_right = np.where(moved_seg_right > 40, 255.0, 0.0)
        mask_super = mask_super.cpu().data.numpy()
        mask_super = np.squeeze(mask_super)
        mask_overlap = mask_overlap.cpu().data.numpy()
        mask_overlap = np.squeeze(mask_overlap)
        x_map = np.sum(mask_overlap, axis=1)
        y_map = np.sum(mask_overlap, axis=0)
        y_l = min(np.argwhere(y_map >= 1.0))[0]
        y_r = max(np.argwhere(y_map >= 1.0))[0]
        x_u = min(np.argwhere(x_map >= 1.0))[0]
        x_d = max(np.argwhere(x_map >= 1.0))[0]

        Dice_score += Dice((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r],
                           (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
        HD_score += cal_Hausdorff((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r],
                                  (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
        NCC_score += window_NCC(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r],
                                (1, 35, 35)).data
        SSIM_score += ssim_loss(left_deformed_data[:, :, x_u: x_d, y_l: y_r],
                                right_deformed_data[:, :, x_u: x_d, y_l: y_r]).item()

        left_deformed_data = left_deformed_data.cpu().data.numpy()
        left_deformed_data = np.squeeze(left_deformed_data)
        right_deformed_data = right_deformed_data.cpu().data.numpy()
        right_deformed_data = np.squeeze(right_deformed_data)
        left_deformed_mask = left_deformed_mask.cpu().data.numpy()
        left_deformed_mask = np.squeeze(left_deformed_mask)
        right_deformed_mask = right_deformed_mask.cpu().data.numpy()
        right_deformed_mask = np.squeeze(right_deformed_mask)
        # left_deformed_data = normalize_img(left_deformed_data)
        # right_deformed_data = normalize_img(right_deformed_data)

        temp_raw_data[index] = right_deformed_data
        temp_deformed_data[index] = left_deformed_data
        temp_seg_left[index] = moved_seg_left
        temp_seg_right[index] = moved_seg_right
        temp_mask_left_data[index] = right_deformed_mask
        temp_mask_right_data[index] = left_deformed_mask

    print(iterate_num)
    print("Dice: ", Dice_score / train_num)
    print("NCC: ", NCC_score / train_num)
    print("Hausdorff: ", HD_score / train_num)
    print("SSIM: ", SSIM_score / train_num)

    with open("%s/%s/iteration/readme" % (data_dir, test_dir), 'a+') as f:
        f.write("iterate_%d: Dice: %.6f; NCC: %.6f; Hausdorff: %.6f; SSIM: %.6f\n"
                % (iterate_num, Dice_score / train_num, NCC_score / train_num, HD_score/train_num, SSIM_score/train_num))

"""
cal time
"""
# test_dir = 'test'
# model_dir = "models/Overlap_prealign_3"
# level_num = 3
#
# model = DualOverlap(level_num)
# model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 30)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("data/test/raw_left.mrc") as mrc:
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     raw_data = standardize_numpy(raw_data)
# with mrcfile.open("data/test/deformed_right.mrc") as mrc:
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# start = time.time()
# for index in tqdm(range(train_num)):
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.1] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#     mask_1, mask_2 = stitch_add_mask_half(raw_mask_cut, transformed_mask_cut)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float().to(device)
#     transformed_cut = torch.Tensor(transformed_cut).float().to(device)
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float().to(device)
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float().to(device)
#     mask_super = torch.Tensor(mask_super).float().to(device)
#     mask_overlap = torch.Tensor(mask_overlap).float().to(device)
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = False
#     model.eval()
#     left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
#         model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)
#
#     left_align = left_align_level[-1]
#     right_align = right_align_level[-1]
#     stn = VariableSpatialTransformer()
#
#     raw_img_cut = torch.Tensor(raw_img_cut).float().to(device)
#     transformed_img_cut = torch.Tensor(transformed_img_cut).float().to(device)
#     raw_img_cut = raw_img_cut.view((-1, 1, *size))
#     transformed_img_cut = transformed_img_cut.view((-1, 1, *size))
#
#     left_deformed_data = stn(transformed_img_cut * transformed_mask_cut, left_align, cpu_flag)
#     right_deformed_data = stn(raw_img_cut * raw_mask_cut, right_align, cpu_flag)
#     left_deformed_data = left_deformed_data.cpu().data.numpy()
#     left_deformed_data = np.squeeze(left_deformed_data)
#     right_deformed_data = right_deformed_data.cpu().data.numpy()
#     right_deformed_data = np.squeeze(right_deformed_data)
#
#     stitching_result = left_deformed_data * mask_2 + right_deformed_data * mask_1
#
# end = time.time()
# print((end - start) / train_num)


"""
cal Hausdorff, SSIM, PSNR
"""
# data_dir = "data"
# test_dir = 'test'
# level_num = 3
# model_dir = "models/Overlap_prealign_NCC_%d" % level_num
#
#
# model = DualOverlap(level_num)
# # model = model.to(device)
#
# model_dir = "%s/%d/net_params_%d.pkl" % (model_dir, level_num+1, 10)
# model.load_state_dict(torch.load(model_dir))
#
# with mrcfile.open("data/test/raw_left.mrc") as mrc:   # 0-1 values
#     raw_data = mrc.data
#     raw_data = raw_data.reshape(-1, 1024, 1024)
#     # raw_data = standardize_numpy(raw_data)
# with mrcfile.open("data/test/deformed_right.mrc") as mrc:   # 0-1 values
#     transformed_data = mrc.data
#     transformed_data = transformed_data.reshape(-1, 1024, 1024)
#     # transformed_data = standardize_numpy(transformed_data)
# with mrcfile.open("data/test/mask_right.mrc") as mrc:
#     transformed_mask_data = mrc.data
#
# seg_left_data = np.load("data/test/raw_left_seg.npy")
# seg_right_data = np.load("data/test/deformed_right_seg.npy")
#
# train_num = raw_data.shape[0]
# img_size = raw_data.shape[1:]
#
# HD_score = 0.0
# SSIM_score = 0.0
# # PSNR_score = 0.0
#
# # print(np.max(raw_data[0]))
# # print(np.max(transformed_data[0]))
#
# ssim_loss = pytorch_ssim.SSIM(window_size=11)
#
# # for index in [0]:
# for index in tqdm(range(train_num)):
#     raw_img = raw_data[index]
#     raw_mask_img = np.zeros(raw_img.shape)
#     raw_mask_img[raw_img > 0.001] = 1
#     transformed_img = transformed_data[index]
#     transformed_mask_img = transformed_mask_data[index]
#     transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
#
#     raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
#                                                                                                          transformed_img,
#                                                                                                          raw_mask_img,
#                                                                                                          transformed_mask_img)
#
#     raw_img_cut = raw_cut.copy()
#     transformed_img_cut = transformed_cut.copy()
#     raw_cut = standardize_numpy(raw_cut)
#     transformed_cut = standardize_numpy(transformed_cut)
#     size = raw_cut.shape
#     raw_cut = torch.Tensor(raw_cut).float()
#     transformed_cut = torch.Tensor(transformed_cut).float()
#     raw_mask_cut = torch.Tensor(raw_mask_cut).float()
#     transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
#     mask_super = torch.Tensor(mask_super).float()
#     mask_overlap = torch.Tensor(mask_overlap).float()
#
#     raw_cut = raw_cut.view((-1, 1, *size))
#     transformed_cut = transformed_cut.view((-1, 1, *size))
#     raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
#     transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
#     mask_super = mask_super.view(-1, 1, *size)
#     mask_overlap = mask_overlap.view(-1, 1, *size)
#
#     cpu_flag = True
#     model.eval()
#     left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
#         model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)
#
#     left_align = left_align_level[-1]
#     right_align = right_align_level[-1]
#     stn = VariableSpatialTransformer()
#
#     raw_img_cut = torch.Tensor(raw_img_cut).float()
#     transformed_img_cut = torch.Tensor(transformed_img_cut).float()
#     raw_img_cut = raw_img_cut.view((-1, 1, *size))
#     transformed_img_cut = transformed_img_cut.view((-1, 1, *size))
#     left_deformed_data = stn(transformed_img_cut * transformed_mask_cut, left_align, cpu_flag)
#     right_deformed_data = stn(raw_img_cut * raw_mask_cut, right_align, cpu_flag)
#
#     mask_super = mask_super.cpu().data.numpy()
#     mask_super = np.squeeze(mask_super)
#     mask_overlap = mask_overlap.cpu().data.numpy()
#     mask_overlap = np.squeeze(mask_overlap)
#     x_map = np.sum(mask_overlap, axis=1)
#     y_map = np.sum(mask_overlap, axis=0)
#     y_l = min(np.argwhere(y_map >= 1.0))[0]
#     y_r = max(np.argwhere(y_map >= 1.0))[0]
#     x_u = min(np.argwhere(x_map >= 1.0))[0]
#     x_d = max(np.argwhere(x_map >= 1.0))[0]
#
#     SSIM_score += ssim_loss(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r]).item()
#
#     # left_deformed_data = left_deformed_data.cpu().data.numpy()
#     # left_deformed_data = np.squeeze(left_deformed_data)
#     # right_deformed_data = right_deformed_data.cpu().data.numpy()
#     # right_deformed_data = np.squeeze(right_deformed_data)
#     #
#     # PSNR_score += psnr(left_deformed_data[x_u: x_d, y_l: y_r], right_deformed_data[x_u: x_d, y_l: y_r])
#
#     seg_slice_left = seg_left_data[index]
#     seg_slice_left = torch.Tensor(seg_slice_left).view(-1, 1, 1024, 1024)
#     seg_slice_right = seg_right_data[index]
#     seg_slice_right = torch.Tensor(seg_slice_right).view(-1, 1, 1024, 1024)
#     moved_seg_right = stn(seg_slice_right, left_align, cpu_flag)
#     moved_seg_left = stn(seg_slice_left, right_align, cpu_flag)
#     moved_seg_left = moved_seg_left.cpu().data.numpy()
#     moved_seg_right = moved_seg_right.cpu().data.numpy()
#     moved_seg_left = np.squeeze(moved_seg_left)
#     moved_seg_right = np.squeeze(moved_seg_right)
#
#     moved_seg_left = np.where(moved_seg_left > 40, 255.0, 0.0)
#     moved_seg_right = np.where(moved_seg_right > 60, 255.0, 0.0)
#
#     # print(cal_Hausdorff((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r]))
#     HD_score += cal_Hausdorff((moved_seg_right * mask_overlap)[x_u: x_d, y_l: y_r], (moved_seg_left * mask_overlap)[x_u: x_d, y_l: y_r])
#
# print("Hausdorff: ", HD_score / train_num)
# print("SSIM: ", SSIM_score / train_num)
# # print("PSNR: ", PSNR_score / train_num)
