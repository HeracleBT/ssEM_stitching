import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from layers import *


class AlignerModuleOverlap(nn.Module):
    """
    Aligner module, accept D_n+1 displacement, output D_n displacement
    """
    def __init__(self, channel):
        super(AlignerModuleOverlap, self).__init__()
        self.residual_estimation = nn.ModuleList()
        feature_maps = [32, 64, 32, 16, 16, 2]
        kernel_size = [7, 7, 7, 7, 3, 3]
        pre_c = channel * 2
        for i in range(len(feature_maps)):
            out_channel = feature_maps[i]
            conv_size = kernel_size[i]
            self.residual_estimation.append(ConvBlock(2, pre_c, out_channel, conv_size, 1, conv_size // 2))
            pre_c = out_channel

    def forward(self, source, target, displace, mask, cpu=False):
        source = source * mask
        target = target * mask
        shape = [*source.shape[2:]]
        displace = F.interpolate(displace, shape, mode='nearest')
        stn = VariableSpatialTransformer()
        deformed = stn(source, displace, cpu)
        residual_input = torch.cat([deformed, target], dim=1)
        residual_output = residual_input
        for layer in self.residual_estimation:
            residual_output = layer(residual_output)
        return (residual_output + displace) * mask


class HierarchicalEncoder(nn.Module):
    def __init__(self, level_num):
        super(HierarchicalEncoder, self).__init__()
        self.level_list = nn.ModuleList()
        for i in range(level_num):
            self.level_list.append(EncoderLayer(8 * i + 1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, source, target, source_mask, target_mask, mask_super):
        source_level = [source]
        target_level = [target]
        source_mask_level = [source_mask]
        target_mask_level = [target_mask]
        mask_super_level = [mask_super]
        for layer in self.level_list:
            mask_super_level.append(self.pool(mask_super_level[-1]))
            source_level.append(layer(source_level[-1]) * mask_super_level[-1])
            target_level.append(layer(target_level[-1]) * mask_super_level[-1])
            source_mask_level.append(self.pool(source_mask_level[-1]))
            target_mask_level.append(self.pool(target_mask_level[-1]))
        return source_level, target_level, source_mask_level, target_mask_level


class HierarchicalOverlap(nn.Module):
    """
    Hierarchical spatial transform estimation
    """

    def __init__(self, level_num):
        super(HierarchicalOverlap, self).__init__()
        self.align_list = nn.ModuleList()
        for i in range(level_num):
            self.align_list.append(AlignerModuleOverlap((level_num - i) * 8 + 1))
        self.align_list.append(AlignerModuleOverlap(1))

    def forward(self, source_level, target_level, source_mask_level, cpu=False):  # left = left pad or not

        displace = torch.zeros((source_level[-1].shape[0], 2, source_level[-1].shape[2] // 2, source_level[-1].shape[3] // 2))
        if not cpu:
            displace = displace.cuda()
        align_level = [displace]

        count = -1
        for layer in self.align_list:
            mask = source_mask_level[count]
            current_source = source_level[count]
            current_target = target_level[count]
            align_level.append(layer(current_source, current_target, align_level[-1], mask, cpu))
            count -= 1
        return align_level


class DualOverlap(nn.Module):

    """
    left_pad: raw[non-overlap] + deformed
    right_pad: raw + deformed[non-overlap]
    left_pad_mask:
    right_pad_mask:
    """

    def __init__(self, level_num):
        super(DualOverlap, self).__init__()
        self.encoder = HierarchicalEncoder(level_num)
        self.left = HierarchicalOverlap(level_num)
        self.right = HierarchicalOverlap(level_num)

    def forward(self, left_pad, right_pad, left_pad_mask, right_pad_mask, mask_super, cpu=False, test=False):
        left_pad_level, right_pad_level, left_pad_mask_level, right_pad_mask_level = self.encoder(left_pad, right_pad, left_pad_mask, right_pad_mask, mask_super)
        left_align_level = self.left(left_pad_level, right_pad_level, left_pad_mask_level, cpu)
        right_align_level = self.right(right_pad_level, left_pad_level, right_pad_mask_level, cpu)
        if not test:
            return left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level
        else:
            return left_align_level, right_align_level, left_pad_mask_level, right_pad_mask_level, left_pad_level, right_pad_level

