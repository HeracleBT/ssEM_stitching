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

if not test_flag:

    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    adjusted_lr_schedule = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    model_dir = "%s/Overlap_prealign_%d" % (model_dir, level_num)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    log_dir = "%s/Overlap_prealign_%d" % (log_dir, level_num)

    if start_epoch > 0:
        pre_model_dir = model_dir
        model.load_state_dict(torch.load("%s/net_params_%d.pkl" % (pre_model_dir, start_epoch)))

    with mrcfile.open(raw_path) as mrc:
        raw_data = mrc.data
        raw_data = raw_data.reshape(-1, 1024, 1024)
        raw_data = standardize_numpy(raw_data)
    with mrcfile.open(deformed_path) as mrc:
        deformed_data = mrc.data
        deformed_data = deformed_data.reshape(-1, 1024, 1024)
        deformed_data = standardize_numpy(deformed_data)
    with mrcfile.open(deformed_mask) as mrc:
        deformed_mask = mrc.data
    train_num = raw_data.shape[0]

    for i in range(-1, -level_num - 2, -1):

        if i == -level_num - 1:
            end_epoch = 40

        for epoch_i in range(start_epoch, end_epoch + 1):
            avg_loss = defaultdict(int)
            model.train()

            index = [i for i in range(train_num)]
            random.shuffle(index)

            accumulate_count = 16
            for k in index:
                raw_img = raw_data[k]
                transformed_img = deformed_data[k]
                transformed_mask_img = deformed_mask[k]
                raw_mask_img = np.zeros((1024, 1024))
                raw_mask_img[300: 812, 300: 812] = 1
                raw_cut, transformed_cut, raw_mask_cut, transformed_mask_cut, mask_super, mask_overlap = pre_process(raw_img, transformed_img, raw_mask_img, transformed_mask_img)
                size = raw_cut.shape

                stn = VariableSpatialTransformer()
                stn = stn.to(device)

                raw_cut = torch.Tensor(raw_cut).float()
                transformed_cut = torch.Tensor(transformed_cut).float()
                raw_mask_cut = torch.Tensor(raw_mask_cut).float()
                transformed_mask_cut = torch.Tensor(transformed_mask_cut).float()
                mask_super = torch.Tensor(mask_super).float()
                mask_overlap = torch.Tensor(mask_overlap).float()

                raw_cut = raw_cut.to(device)
                transformed_cut = transformed_cut.to(device)
                raw_mask_cut = raw_mask_cut.to(device)
                transformed_mask_cut = transformed_mask_cut.to(device)
                mask_super = mask_super.to(device)
                mask_overlap = mask_overlap.to(device)
                raw_cut = raw_cut.view((-1, 1, *size))
                transformed_cut = transformed_cut.view((-1, 1, *size))
                raw_mask_cut = raw_mask_cut.view((-1, 1, *size))
                transformed_mask_cut = transformed_mask_cut.view((-1, 1, *size))
                mask_super = mask_super.view(-1, 1, *size)
                mask_overlap = mask_overlap.view(-1, 1, *size)
                left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level = model(transformed_cut,
                                                                                                       raw_cut,
                                                                                                       transformed_mask_cut,
                                                                                                       raw_mask_cut,
                                                                                                       mask_super)

                left_align = left_align_level[-i]
                right_align = right_align_level[-i]

                left_up_align = F.interpolate(left_align, transformed_cut.shape[2:],
                                              mode='nearest') if i != -level_num - 1 else left_align
                right_up_align = F.interpolate(right_align, raw_cut.shape[2:],
                                               mode='nearest') if i != -level_num - 1 else right_align

                left_deformed_data = stn(transformed_cut * transformed_mask_cut, left_up_align)
                right_deformed_data = stn(raw_cut * raw_mask_cut, right_up_align)
                loss_discrepancy = torch.sum(torch.square(left_deformed_data * mask_overlap - right_deformed_data * mask_overlap)) / torch.sum(mask_overlap)

                tmp_y = torch.zeros(left_align.shape)
                tmp_y = tmp_y.to(device)
                tmp_x = torch.zeros(left_align.shape)
                tmp_x = tmp_x.to(device)
                tmp_y[:, :, 1:, :] = left_align[:, :, :-1, :]
                tmp_x[:, :, :, 1:] = left_align[:, :, :, :-1]
                x_x = left_align - tmp_x
                x_x[:, :, :, :1] = 0.0
                x_y = left_align - tmp_y

                rtmp_y = torch.zeros(right_align.shape)
                rtmp_y = rtmp_y.to(device)
                rtmp_x = torch.zeros(right_align.shape)
                rtmp_x = rtmp_x.to(device)
                rtmp_y[:, :, 1:, :] = right_align[:, :, :-1, :]
                rtmp_x[:, :, :, 1:] = right_align[:, :, :, :-1]
                rx_x = right_align - rtmp_x
                rx_x[:, :, :, :1] = 0.0
                rx_y = right_align - rtmp_y

                loss_reg = torch.mean(torch.square(x_x)) + torch.mean(torch.square(x_y)) + torch.mean(
                    torch.square(rx_x)) + torch.mean(torch.square(rx_y))
                gamma_1 = torch.Tensor([0.2]).to(device)

                jaco_align = left_align.permute(0, 2, 3, 1)
                jaco_align = jacobian_determinant(jaco_align)
                rjaco_align = right_align.permute(0, 2, 3, 1)
                rjaco_align = jacobian_determinant(rjaco_align)
                loss_jaco = torch.mean(F.relu(jaco_align) - jaco_align) + torch.mean(F.relu(rjaco_align) - rjaco_align)
                gamma_2 = torch.Tensor([50.0]).to(device)
                loss_all = loss_discrepancy + torch.mul(gamma_1, loss_reg) + torch.mul(gamma_2, loss_jaco)

                loss_all.backward()

                if (i + 1) % accumulate_count == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                assert loss_discrepancy.item()
                output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f, Reg Loss: %.4f,  Jaco Loss: %.4f\n" % \
                              (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item(), loss_reg.item(),
                               loss_jaco.item())

                print(output_data)
                avg_loss['loss_all'] += loss_all.item()
                avg_loss['loss_discrepancy'] += loss_discrepancy.item()
                avg_loss['loss_reg'] += loss_reg.item()
                avg_loss['loss_jaco'] += loss_jaco.item()

            for k, v in avg_loss.items():
                avg_loss[k] = (v * batch_size) / train_num

            log_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f,  Reg Loss: %.4f,  Jaco Loss: %.4f\n" % \
                       (epoch_i, end_epoch, avg_loss['loss_all'], avg_loss['loss_discrepancy'], avg_loss['loss_reg'],
                        avg_loss['loss_jaco'])
            log_file = open(log_dir, 'a')
            log_file.write(log_data)
            log_file.close()

            for k, v in avg_loss.items():
                avg_loss[k] = 0

            stage_dir = "%s/%d" % (model_dir, -i)
            if not os.path.exists(stage_dir):
                os.mkdir(stage_dir)

            torch.save(model.state_dict(), "%s/net_params_%d.pkl" % (stage_dir, epoch_i))

            adjusted_lr_schedule.step()

        del stn

else:

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
    # mask_1, mask_2 = stitch_add_mask_linear(raw_mask_cut, transformed_mask_cut)
    # mask_1, mask_2 = stitch_add_mask_half(raw_mask_cut, transformed_mask_cut)
    # cv2.imshow("mask1", mask_1)
    # cv2.imshow("mask2", mask_2)
    init_stitching_result = normalize_img(raw_cut * raw_mask_cut) + normalize_img(
        transformed_cut * transformed_mask_cut)
    init_mask_result = raw_mask_cut + transformed_mask_cut
    # mask_result = left_deformed_mask + raw_mask_cut
    init_mask_result[init_mask_result == 0] = np.nan
    init_stitching_result = init_stitching_result / init_mask_result
    # cv2.imshow("init_stitch", init_stitching_result)

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

    # layer_v = -5
    # left_align = left_align_level[layer_v]
    # right_align = right_align_level[layer_v]
    #
    # temp_align = left_align
    # mesh_len = left_align.shape[-1]
    # # print(mesh_len)
    # x, y = np.meshgrid(np.linspace(-1, 1, mesh_len), np.linspace(-1, 1, mesh_len))
    # temp_align = temp_align.cpu().data.numpy()
    # temp_align = np.squeeze(temp_align)
    # print(temp_align.shape)
    # # right_align = right_align.cpu().data.numpy()
    # interval = 5
    # l_start = 300 // np.abs(layer_v)
    # r_end = 812 // np.abs(layer_v)
    # plt.quiver(x[l_start:r_end:interval, l_start:r_end:interval], y[l_start:r_end:interval, l_start:r_end:interval],
    #            (x + temp_align[0])[l_start:r_end:interval, l_start:r_end:interval], (y + temp_align[1])[l_start:r_end:interval, l_start:r_end:interval])
    # ax = plt.gca()  # 获取到当前坐标轴信息
    # ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    # ax.invert_yaxis()  # 反转Y坐标轴
    # plt.show()

    # print(len(left_pad_level))
    # for i in range(5):
    #     # print(left_pad_level[i].shape)
    #     left_pad_img = right_pad_level[i][0]
    #     left_pad_img = left_pad_img.cpu().data.numpy()
    #     left_pad_img = np.sum(left_pad_img, axis=0)
    #     left_pad_img = normalize_img(left_pad_img)
    #     cv2.imshow(str(i), left_pad_img)

    left_align = left_align_level[-1]
    right_align = right_align_level[-1]

    stn = VariableSpatialTransformer()
    # stn = stn.to(device)
    m_stn = VariableSpatialTransformer(mode='nearest')
    # m_stn = m_stn.to(device)

    # mesh_grid_left = np.zeros((1024, 1024))
    # mesh_grid_right = cv2.imread("%s/%s/mesh_grid_right.png" % (data_dir, test_dir), cv2.IMREAD_GRAYSCALE)
    # mesh_grid_right = np.where(mesh_grid_right > 0.1, 1.0, 0)
    # for i in range(300, 810, 50):
    #     mesh_grid_left[i:i + 5, 300: 812] = 1.0
    #     mesh_grid_left[300: 812, i:i + 5] = 1.0
    #
    # mesh_grid_left = torch.Tensor(mesh_grid_left).float()
    # mesh_grid_right = torch.Tensor(mesh_grid_right).float()
    # mesh_grid_left = mesh_grid_left.view(1, 1, 1024, 1024)
    # mesh_grid_right = mesh_grid_right.view(1, 1, 1024, 1024)
    #
    # mesh_grid_left = stn(mesh_grid_left, right_align, cpu_flag)
    # mesh_grid_right = stn(mesh_grid_right, left_align, cpu_flag)
    # mesh_grid_left = mesh_grid_left.cpu().data.numpy()
    # mesh_grid_left = np.squeeze(mesh_grid_left)
    # mesh_grid_left = np.where(mesh_grid_left > 0.05, 1.0, 0.0)
    # mesh_grid_right = mesh_grid_right.cpu().data.numpy()
    # mesh_grid_right = np.squeeze(mesh_grid_right)
    # mesh_grid_right = np.where(mesh_grid_right > 0.05, 1.0, 0.0)
    #
    # mesh_grid_left = np.stack([np.zeros((1024, 1024)), np.zeros((1024, 1024)), mesh_grid_left], axis=2)
    # mesh_grid_right = np.stack([mesh_grid_right, np.zeros((1024, 1024)), np.zeros((1024, 1024))], axis=2)
    # # cv2.imwrite("data/stitching/detailed_test/images/%d/panorama_warp_left.png" % index, mesh_grid_left)
    # # cv2.imwrite("data/stitching/detailed_test/images/%d/panorama_warp_right.png" % index, mesh_grid_right)
    # cv2.imshow('warp_left', mesh_grid_left)
    # cv2.imshow('warp_right', mesh_grid_right)

    # lll = cv2.imread("%s/%s/warp_left_our.png" % (data_dir, test_dir))
    # rrr = cv2.imread("%s/%s/warp_right_our.png" % (data_dir, test_dir))
    # lll = cv2.cvtColor(lll, cv2.COLOR_BGR2BGRA)
    # rrr = cv2.cvtColor(rrr, cv2.COLOR_BGR2BGRA)
    # w, h, c = lll.shape
    # for i in range(w):
    #     for j in range(h):
    #         b, g, r, _ = lll[i, j]
    #         if r == 0 and g == 0 and b == 0:
    #             lll[i][j] = [0, 0, 0, 0]
    #         b, g, r, _ = rrr[i, j]
    #         if r == 0 and g == 0 and b == 0:
    #             rrr[i][j] = [0, 0, 0, 0]
    # cv2.imwrite("%s/%s/warp_left_our.png" % (data_dir, test_dir), lll)
    # cv2.imwrite("%s/%s/warp_right_our.png" % (data_dir, test_dir), rrr)

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
    x_map = np.sum(mask_overlap, axis=1)
    y_map = np.sum(mask_overlap, axis=0)
    y_l = min(np.argwhere(y_map >= 1.0))[0]
    y_r = max(np.argwhere(y_map >= 1.0))[0]
    x_u = min(np.argwhere(x_map >= 1.0))[0]
    x_d = max(np.argwhere(x_map >= 1.0))[0]

    print(window_NCC(left_deformed_data[:, :, x_u: x_d, y_l: y_r], right_deformed_data[:, :, x_u: x_d, y_l: y_r],
                     (1, 35, 35)).data)
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    print(ssim_loss(left_deformed_data[:, :, x_u: x_d, y_l: y_r],
                                right_deformed_data[:, :, x_u: x_d, y_l: y_r]).item())

    # left_deformed_data = left_deformed_data.cpu().data.numpy()
    # left_deformed_data = np.squeeze(left_deformed_data)
    # right_deformed_data = right_deformed_data.cpu().data.numpy()
    # right_deformed_data = np.squeeze(right_deformed_data)
    # left_deformed_mask = left_deformed_mask.cpu().data.numpy()
    # left_deformed_mask = np.squeeze(left_deformed_mask)
    # right_deformed_mask = right_deformed_mask.cpu().data.numpy()
    # right_deformed_mask = np.squeeze(right_deformed_mask)
    # # left_deformed_data = normalize_img(left_deformed_data)
    # # right_deformed_data = normalize_img(right_deformed_data)
    #
    # # stitching_result = left_deformed_data + right_deformed_data
    # # mask_result = left_deformed_mask + right_deformed_mask
    # # mask_result[mask_result == 0] = np.nan
    # # stitching_result = stitching_result / mask_result
    #
    # stitching_result = left_deformed_data * mask_2 + right_deformed_data * mask_1
    # # stitching_result = normalize_img(stitching_result)
    # stitching_result = np.uint8(stitching_result)
    #
    # mass = mask_1 + mask_2
    # cv2.imshow("mask_super", mass)
    #
    # cv2.imshow("left_deformed_data", left_deformed_data)
    # cv2.imshow("right_deformed_data", right_deformed_data)
    # cv2.imshow("left_deformed_mask", left_deformed_mask)
    # cv2.imshow("right_deformed_mask", right_deformed_mask)
    # cv2.imshow("stitching_result", stitching_result)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
