import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from sync_batchnorm.batchnorm import SynchronizedBatchNorm1d


class ChannelBlock(nn.Module):
    def __init__(self, channel_num, mode='No-residual'):
        super(ChannelBlock, self).__init__()
        self.mode = mode
        self.channel_num = channel_num
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(self.channel_num, int(self.channel_num/16))
        self.linear2 = nn.Linear(int(self.channel_num/16), self.channel_num)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.pool(x)  # shape = (b, c, 1, 1)
        x1 = x1.squeeze(dim=3).squeeze(dim=2)   # shape = (b, c)
        x2 = self.linear1(x1)
        x2 = self.relu(x2)
        x2 = self.linear2(x2)
        x2 = self.sigmoid(x2)
        x3 = x2.unsqueeze(dim=2).unsqueeze(dim=3)
        if self.mode == 'residual':
            return x * x3 + x
        else:
            return x * x3


class ChannelBlock1(nn.Module):
    def __init__(self, channel_num, mode='residual'):
        super(ChannelBlock1, self).__init__()
        self.mode = mode
        self.channel_num = channel_num
        self.conv = nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, padding=1)

    def forward(self, x):
        [b, c, h, w] = x.size()
        x = self.conv(x)
        x_cxhw = x.view(b, c, h * w)
        x_hwxc = x_cxhw.permute(0, 2, 1)
        x_cxc = torch.bmm(x_cxhw, x_hwxc)
        out = torch.bmm(x_hwxc, torch.softmax(x_cxc, dim=1))
        out = out.view(b, c, h, w)
        if self.mode == 'residual':
            return out + x
        else:
            return out


class ChannelBlock2(nn.Module):
    def __init__(self, channel_num, mode='residual'):
        super(ChannelBlock2, self).__init__()
        self.mode = mode
        self.channel_num = channel_num
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(nn.Conv1d(1, 1, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv1d(1, 1, kernel_size=7, padding=3, stride=2),
                                  SynchronizedBatchNorm1d(1),
                                  nn.ReLU(),
                                  nn.ConvTranspose1d(1, 1, kernel_size=2, stride=2),
                                  SynchronizedBatchNorm1d(1),
                                  nn.Sigmoid())

    def forward(self, x):
        [b, c, h, w] = x.shape
        x1 = self.pool(x) # b, c, 1, 1
        x2 = x1.view(b, 1, c)
        x3 = self.conv(x2)
        #x3 = torch.sigmoid(x3)
        x3 = x3.view(b, c, 1, 1)
        if self.mode == 'residual':
            return x * x3 + x
        else:
            return x * x3



class PointBlock(nn.Module):
    def __init__(self, channel_num, mode='residual'):
        super(PointBlock, self).__init__()
        self.mode = mode
        self.channel_num = channel_num
        self.conv = nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, padding=1)

    def forward(self, x):
        [b, c, h, w] = x.size()
        x = self.conv(x)
        x_cxhw = x.view(b, c, h * w)
        x_hwxc = x_cxhw.permute(0, 2, 1)
        x_hwxhw = torch.bmm(x_hwxc, x_cxhw)
        out = torch.bmm(x_cxhw, torch.softmax(x_hwxhw, dim=1))
        out = out.view(b, c, h, w)
        if self.mode == 'residual':
            return out + x
        else:
            return out


class PCblock(nn.Module):
    def __init__(self, channel_num):
        super(PCblock, self).__init__()
        self.channel_num = channel_num
        self.channel_att = ChannelBlock(self.channel_num, mode='No-residual')
        self.point_att = PointBlock(self.channel_num, mode='residual')

    def forward(self, x):
        x1 = self.channel_att(x)
        x2 = self.point_att(x)
        out = x1 + x2
        return out


class PCblock_Cat(nn.Module):
    def __init__(self, channel_num):
        super(PCblock_Cat, self).__init__()
        self.channel_num = channel_num
        self.channel_att = ChannelBlock(self.channel_num, mode='No-residual')
        self.point_att = PointBlock(self.channel_num, mode='No-residual')
        self.conv = nn.Conv2d(self.channel_num * 2, self.channel_num, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.channel_att(x)
        x2 = self.point_att(x)
        out = torch.cat((x1, x2), 1)
        out = self.conv(out)
        return out + x


class PCblock1(nn.Module):
    def __init__(self, channel_num, mode='residual'):
        super(PCblock1, self).__init__()
        self.mode = mode
        self.channel_num = channel_num
        self.conv = nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, padding=1)
        self.channel_attention = ChannelBlock(self.channel_num, mode='No-residual')

    def forward(self, x):
        [b, c, h, w] = x.size()
        x = self.conv(x)
        x_cxhw = x.view(b, c, h * w)
        x_hwxc = x_cxhw.permute(0, 2, 1)
        x_hwxhw = torch.bmm(x_hwxc, x_cxhw)
        out = torch.bmm(self.channel_attention(x).view(b, c, h * w), torch.softmax(x_hwxhw, dim=1))
        out = out.view(b, c, h, w)
        if self.mode == 'residual':
            return out + x
        else:
            return out


class PCblock2(nn.Module):
    def __init__(self, channel_num, mode='residual'):
        super(PCblock2, self).__init__()
        self.mode = mode
        self.channel_num = channel_num
        self.conv = nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, padding=1)
        self.channel_attention = ChannelBlock2(self.channel_num, mode='No-residual')
        self.point_attention = nn.Softmax(dim=1)

    def forward(self, x):
        [b, c, h, w] = x.size()
        x = self.conv(x)
        x_cxhw = x.view(b, c, h * w)
        x_hwxc = x_cxhw.permute(0, 2, 1)
        x_hwxhw = self.point_attention(torch.bmm(x_hwxc, x_cxhw))
        # out = torch.bmm(self.channel_attention(x).view(b, c, h * w), torch.softmax(x_hwxhw, dim=1))
        out = torch.bmm(self.channel_attention(x).view(b, c, h * w), x_hwxhw)
        out = out.view(b, c, h, w)
        if self.mode == 'residual':
            return out + x
        else:
            return out


class PointBlock2(nn.Module):
    def __init__(self, channel_num, mode='residual'):
        super(PointBlock2, self).__init__()
        self.mode = mode
        self.channel_num = channel_num
        self.conv = nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, padding=1)

    def forward(self, x):
        [b, c, h, w] = x.size()
        x = self.conv(x)
        x_cxhw = x.view(b, c, h * w)
        x_hwxc = x_cxhw.permute(0, 2, 1)
        x_hwxhw = torch.bmm(x_hwxc, x_cxhw)
        
        return torch.softmax(x_hwxhw, dim=1)


class PCblock_ori(nn.Module):
    def __init__(self, channel_num):
        super(PCblock_ori, self).__init__()
        self.channel_num = channel_num
        self.channel_att = ChannelBlock1(self.channel_num, mode='No-residual')
        self.point_att = PointBlock(self.channel_num, mode='No-residual')
        self.conv = nn.Conv2d(2 * channel_num, channel_num, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.channel_att(x)
        x2 = self.point_att(x)
        out = torch.cat((x1, x2), dim=1)
        out = self.conv(out)
        return out + x
"""
def attention_map(A, fold):
    b, c, h, w = A.shape
    attention_map = torch.zeros(b, c, h*w, dtype=torch.float64)
    # sum_column = sum_softmax(A, int(fold))

    for j in range(fold):
        for i in range(fold):
            b_column = A.reshape(b, c, h * w).permute(0, 2, 1)[:, i * int(h * w / fold): (i + 1) * int(h * w / fold), :]
            b_raw = A.reshape(b, c, h * w)[:, :, j * int(h * w / fold): (j + 1) * int(h * w / fold)]
            sub_column = torch.bmm(b_column, b_raw)
            if i == 0:
                sub_column_to_cat = sub_column
            else:
                sub_column_to_cat = torch.cat((sub_column_to_cat, sub_column), dim=1)

        for n in range(b):
            d = torch.mm(A.reshape(b, c, h * w)[n, :, :], torch.softmax(sub_column_to_cat[n, :, :], dim=0))
            # attention_map[n, :, int(j * h * w / fold): int((j + 1) * h * w / fold)] = attention_map[n, :, int(j * h * w / fold): int((j + 1) * h * w / fold)] + d.cpu()
            attention_map[n, :, int(j * h * w / fold): int((j + 1) * h * w / fold)] = d.cpu()

    return attention_map.view(b, c, h, w).cuda()
"""
"""
# class ChannelBlock(nn.Module):
#     def __init__(self, channel_num, mode='No-residual'):
#         super(ChannelBlock, self).__init__()
#         self.mode = mode
#         self.channel_num = channel_num
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.linear = nn.Linear(self.channel_num, self.channel_num)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x1 = self.pool(x)  # shape = (b, c, 1, 1)
#         x1 = torch.squeeze(x1, 2).permute(0, 2, 1)  # shape = (b, 1, c)
#         x2 = self.linear(x1)  # shape = (b, 1, c)
#         x2 = self.relu(x2)  # shape = (b, 1, c)
#         x3 = torch.unsqueeze(x2.permute(0, 2, 1), 2)  # shape = (b, c, 1, 1)
#         if self.mode == 'residual':
#             return x * x3 + x
#         else:
#             return x * x3
"""


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = SynchronizedBatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # # Initialise weights
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g, self.theta, self.phi, self.psi, self.W)
        return output

    def _concatenation(self, x, g, theta, phi, psi, W):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, h, w) -> (b, i_c, h, w) -> (b, i_c, hw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, h', w') -> phi_g (b, i_c, h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, hw) -> (b, i_c, h/s1, w/s2)
        phi_g = F.interpolate(phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = W(y)

        return W_y, sigm_psi_f


    # def _concatenation(self, x, g):
    #     input_size = x.size()
    #     batch_size = input_size[0]
    #     assert batch_size == g.size(0)
    #
    #     # theta => (b, c, h, w) -> (b, i_c, h, w) -> (b, i_c, hw)
    #     # phi   => (b, g_d) -> (b, i_c)
    #     theta_x = self.theta(x)
    #     theta_x_size = theta_x.size()
    #
    #     # g (b, c, h', w') -> phi_g (b, i_c, h', w')
    #     #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, hw) -> (b, i_c, h/s1, w/s2)
    #     phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=True)
    #     f = F.relu(theta_x + phi_g, inplace=True)
    #
    #     #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
    #     sigm_psi_f = torch.sigmoid(self.psi(f))
    #
    #     # upsample the attentions and multiply
    #     sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
    #     y = sigm_psi_f.expand_as(x) * x
    #     W_y = self.W(y)
    #
    #     return W_y, sigm_psi_f


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2, mode='concatenation',
                 sub_sample_factor=(2, 2)):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2, mode=mode,
                                                   sub_sample_factor=sub_sample_factor
                                                   )

class SE_Module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
