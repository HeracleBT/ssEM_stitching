import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from layers import *


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):
        # print(x.shape)
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x


class VxmDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, source_mask, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field * source_mask
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return y_source, pos_flow
        else:
            return y_source, pos_flow


class Unet_s(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x, source_mask):
        # print(x.shape)
        # get encoder activations
        x_enc = [x]
        mask_enc = [source_mask]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
            mask_enc.append(self.pool(mask_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        source_mask = mask_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            source_mask = mask_enc.pop()
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x


class VxmDense_s(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_s(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, source_mask, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x, source_mask)

        x = x * source_mask
        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return y_source, pos_flow
        else:
            return y_source, pos_flow


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
        # inital_mask = source_mask_level[-1]
        # x_u = torch.min(torch.nonzero(inital_mask[0, 0, :, :]), dim=0)[0][0]
        # y_l = torch.min(torch.nonzero(inital_mask[0, 0, :, :]), dim=0)[0][1]
        # x_d = torch.max(torch.nonzero(inital_mask[0, 0, :, :]), dim=0)[0][0]
        # y_r = torch.max(torch.nonzero(inital_mask[0, 0, :, :]), dim=0)[0][1]
        # inital_mask = inital_mask[:, :, x_u:x_d, y_l:y_r]
        # vectors = [torch.arange(0, i // 2) for i in [inital_mask.shape[2], inital_mask.shape[3]]]
        # grids = torch.meshgrid(vectors)
        # grid = torch.stack(grids)
        # grid = torch.unsqueeze(grid, 0)
        # grid = grid.type(torch.FloatTensor).cuda()
        # for i in range(2):
        #     grid[:, i, ...] = 2 * (grid[:, i, ...] / (inital_mask.shape[2 + i] - 1) - 0.5)
        # grid = grid.repeat(source_level[-1].shape[0], 1, 1, 1)
        # align_level = [grid]

        # vectors = [torch.arange(0, i // 2) for i in [source_level[-1].shape[2], source_level[-1].shape[3]]]
        # grids = torch.meshgrid(vectors)
        # grid = torch.stack(grids)
        # grid = torch.unsqueeze(grid, 0)
        # grid = grid.type(torch.FloatTensor).cuda()
        # for i in range(2):
        #     grid[:, i, ...] = 2 * (grid[:, i, ...] / (source_level[-1].shape[2 + i] - 1) - 0.5)
        # grid = grid.repeat(source_level[-1].shape[0], 1, 1, 1)
        # align_level = [grid]

        displace = torch.zeros((source_level[-1].shape[0], 2, source_level[-1].shape[2] // 2, source_level[-1].shape[3] // 2))
        if not cpu:
            displace = displace.cuda()
        align_level = [displace]

        count = -1
        for layer in self.align_list:
            mask = source_mask_level[count]
            current_source = source_level[count]
            current_target = target_level[count]
            # x_u = torch.min(torch.nonzero(mask[0, 0, :, :]), dim=0)[0][0]
            # y_l = torch.min(torch.nonzero(mask[0, 0, :, :]), dim=0)[0][1]
            # x_d = torch.max(torch.nonzero(mask[0, 0, :, :]), dim=0)[0][0]
            # y_r = torch.max(torch.nonzero(mask[0, 0, :, :]), dim=0)[0][1]
            # current_source = current_source[:, :, x_u:x_d, y_l:y_r]
            # current_target = current_target[:, :, x_u:x_d, y_l:y_r]
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


class UniOverlap(nn.Module):
    """
        left_pad: raw[non-overlap] + deformed
        right_pad: raw + deformed[non-overlap]
        left_pad_mask:
        right_pad_mask:
        """

    def __init__(self, level_num):
        super(UniOverlap, self).__init__()
        self.encoder = HierarchicalEncoder(level_num)
        self.left = HierarchicalOverlap(level_num)

    def forward(self, left_pad, right_pad, left_pad_mask, right_pad_mask, mask_super, cpu=False):
        left_pad_level, right_pad_level, left_pad_mask_level, right_pad_mask_level = self.encoder(left_pad, right_pad,
                                                                                                  left_pad_mask,
                                                                                                  right_pad_mask,
                                                                                                  mask_super)
        left_align_level = self.left(left_pad_level, right_pad_level, left_pad_mask_level, cpu)
        return left_align_level, left_pad_mask_level

