import numpy as np
import cv2
import os
import torch
from torch.optim import lr_scheduler
from networks import *
from layers import SpatialTransformer
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import mrcfile
import random
import pytorch_ssim
from argparse import ArgumentParser
from collections import defaultdict
from Utils import standardize_numpy, jacobian_determinant, pre_process, normalize_img, stitch_add_mask_linear, \
    stitch_add_mask_half, window_NCC, stitch_add_mask_linear_border
import matplotlib.pyplot as plt

parser = ArgumentParser(description="Fast and Accurate")

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=10, help='epoch number of end training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--level', type=int, default=3, help='level num')
parser.add_argument('--batch_size', type=int, default=12, help='batch size')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--model_dir', type=str, default='models', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--train_dir', type=str, default='train', help='training data name')
parser.add_argument('--test_dir', type=str, default='test', help='testing data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--test', action='store_true', default=False, help='train or test')

args = parser.parse_args()
start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
gpu_list = args.gpu_list
model_dir = args.model_dir
data_dir = args.data_dir
log_dir = args.log_dir
test_dir = args.test_dir
train_dir = args.train_dir
test_flag = args.test
level_num = args.level
batch_size = args.batch_size

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0")

raw_path = "%s/%s/raw_left.mrc" % (data_dir, train_dir)
deformed_path = "%s/%s/deformed_right.mrc" % (data_dir, train_dir)
deformed_mask = "%s/%s/mask_right.mrc" % (data_dir, train_dir)

model = DualOverlap(level_num)
# model = nn.DataParallel(model)
# model = model.to(device)

test_dir = 'test'
raw_path = "%s/%s/raw_left.png" % (data_dir, test_dir)
deformed_path = "%s/%s/deformed_right.png" % (data_dir, test_dir)
raw_mask = "%s/%s/mask_left.png" % (data_dir, test_dir)
deformed_mask = "%s/%s/mask_right.png" % (data_dir, test_dir)
model_dir = "%s/Overlap_prealign_new_%d/%d/net_params_%d.pkl" % (model_dir, level_num, level_num + 1, 20)
# model_dir = "%s/Overlap_prealign_%d/%d/net_params_%d.pkl" % (model_dir, level_num, level_num + 1, 25)
model.load_state_dict(torch.load(model_dir))
# model = model.to(device)
model.eval()

raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
transformed_img = cv2.imread(deformed_path, cv2.IMREAD_GRAYSCALE)
transformed_mask_img = cv2.imread(deformed_mask, cv2.IMREAD_GRAYSCALE)
raw_mask_img = cv2.imread(raw_mask, cv2.IMREAD_GRAYSCALE)

transformed_mask_img = np.where(transformed_mask_img > 0, 1.0, 0)
raw_mask_img = np.where(raw_mask_img > 0, 1.0, 0)

raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img,
                                                                                                        transformed_img,
                                                                                                        raw_mask_img,
                                                                                                        transformed_mask_img)
raw_img_cut = raw_cut.copy()
transformed_img_cut = transformed_cut.copy()
raw_cut = standardize_numpy(raw_cut)
transformed_cut = standardize_numpy(transformed_cut)

mask_1, mask_2 = stitch_add_mask_linear_border(raw_mask_cut, transformed_mask_cut, mode="r")
init_stitching_result = normalize_img(raw_cut * raw_mask_cut) + normalize_img(
    transformed_cut * transformed_mask_cut)
init_mask_result = raw_mask_cut + transformed_mask_cut
init_mask_result[init_mask_result == 0] = np.nan
init_stitching_result = init_stitching_result / init_mask_result

size = raw_cut.shape
raw_cut = torch.Tensor(raw_cut).float()
transformed_cut = torch.Tensor(transformed_cut).float()
raw_mask_cut = torch.Tensor(raw_mask_cut).float()
transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
mask_super = torch.Tensor(mask_super).float()
mask_overlap = torch.Tensor(mask_overlap).float()

# raw_cut = raw_cut.to(device)
# transformed_cut = transformed_cut.to(device)
# raw_mask_cut = raw_mask_cut.to(device)
# transformed_mask_cut = transformed_mask_cut.to(device)
# mask_super = mask_super.to(device)
# mask_overlap = mask_overlap.to(device)

raw_cut = raw_cut.view((-1, 1, *size))
transformed_cut = transformed_cut.view((-1, 1, *size))
raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
mask_super = mask_super.view(-1, 1, *size)
mask_overlap = mask_overlap.view(-1, 1, *size)

cpu_flag = True
left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level = \
    model(transformed_cut, raw_cut, transformed_mask_cut, raw_mask_cut, mask_super, cpu_flag, test=True)

left_align = left_align_level[-1]
right_align = right_align_level[-1]

stn = VariableSpatialTransformer()
# stn = stn.to(device)
m_stn = VariableSpatialTransformer(mode='nearest')

raw_img_cut = torch.Tensor(raw_img_cut).float()
transformed_img_cut = torch.Tensor(transformed_img_cut).float()
raw_img_cut = raw_img_cut.view((-1, 1, *size))
transformed_img_cut = transformed_img_cut.view((-1, 1, *size))

left_deformed_data = stn(transformed_img_cut * transformed_mask_cut, left_align, cpu_flag)
right_deformed_data = stn(raw_img_cut * raw_mask_cut, right_align, cpu_flag)
left_deformed_mask = m_stn(transformed_mask_cut, left_align, cpu_flag)
right_deformed_mask = m_stn(raw_mask_cut, right_align, cpu_flag)

mask_overlap = mask_overlap.cpu().data.numpy()
mask_overlap = np.squeeze(mask_overlap)

left_deformed_data = left_deformed_data.cpu().data.numpy()
left_deformed_data = np.squeeze(left_deformed_data)
right_deformed_data = right_deformed_data.cpu().data.numpy()
right_deformed_data = np.squeeze(right_deformed_data)
left_deformed_mask = left_deformed_mask.cpu().data.numpy()
left_deformed_mask = np.squeeze(left_deformed_mask)
right_deformed_mask = right_deformed_mask.cpu().data.numpy()
right_deformed_mask = np.squeeze(right_deformed_mask)

stitching_result = left_deformed_data * mask_2 + right_deformed_data * mask_1
stitching_result = np.uint8(stitching_result)
