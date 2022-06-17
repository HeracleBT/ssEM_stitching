import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import channel_normalize, patch_mean, patch_std


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=None):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2)
        BN = getattr(nn, 'BatchNorm%dd' % ndims)
        IN = getattr(nn, 'InstanceNorm%dd' % ndims)
        if norm == 'batch':
            self.n = BN(out_channels)
        elif norm == 'instance':
            self.n = IN(out_channels)
        else:
            self.n = None

    def forward(self, x):
        out = self.main(x)
        if self.n:
            out = self.n(out)
        out = self.activation(out)
        return out


class VariableSpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, mode='bilinear'):
        super().__init__()

        self.mode = mode

    def forward(self, src, flow, cpu=False):

        size = src.shape[2:]
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        if not cpu:
            grid = grid.cuda()
        new_locs = grid + flow

        # grid = self.grid.cpu()
        # new_locs = grid + flow

        shape = flow.shape[2:]
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class EncoderLayer(nn.Module):

    def __init__(self, channel):
        super(EncoderLayer, self).__init__()
        self.input_c = channel
        self.conv_1 = ConvBlock(2, self.input_c, self.input_c + 4, norm='batch')
        self.conv_2 = ConvBlock(2, self.input_c + 4, self.input_c + 8, norm='batch')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.pool(x)
        return x
