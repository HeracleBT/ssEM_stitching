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


class ConvTransposeBlock(nn.Module):
    """
        Specific transposed convolutional block followed by leakyrelu for unet.
        """

    def __init__(self, ndims, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0, batch_norm=False):
        super().__init__()

        Conv = getattr(nn, 'ConvTranspose%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activation = nn.LeakyReLU(0.2)
        BN = getattr(nn, 'BatchNorm%dd' % ndims)
        if batch_norm:
            self.bn = BN(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        out = self.main(x)
        if self.bn:
            out = self.bn(out)
        out = self.activation(out)
        return out


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # self.size = src.shape[2:]
        # vectors = [torch.arange(0, s) for s in self.size]
        # grids = torch.meshgrid(vectors)
        # grid = torch.stack(grids)
        # grid = torch.unsqueeze(grid, 0)
        # grid = grid.type(torch.FloatTensor)
        # new_locs = grid + flow

        # grid = self.grid.cpu()
        # new_locs = grid + flow

        # new locations
        new_locs = self.grid + flow
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


class ImageGradientPenalize(nn.Module):
    """
    Gradient calculate for images
    """

    def __init__(self):
        super(ImageGradientPenalize, self).__init__()

    def forward(self, x):
        tmp_y = torch.zeros(x.shape)
        tmp_x = torch.zeros(x.shape)
        tmp_y[:, :, 1:, :] = x[:, :, :-1, :]
        tmp_x[:, :, :, 1:] = x[:, :, :, :-1]
        x_x = x - tmp_x
        x_x[:, :, :, :1] = 0.0
        x_y = x - tmp_y
        x_y[:, :, :1, :] = 0.0
        gradient = torch.stack([x_x, x_y], dim=0)

        indicator = gradient + 1.0
        indicator = torch.where(indicator > 0, torch.full_like(indicator, 0), torch.abs(indicator))
        return indicator * torch.square(gradient)


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


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x
