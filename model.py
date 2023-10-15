import torch
from torch import nn
import torchvision.models as models
import numpy as np
from torch.nn.init import trunc_normal_
import ops
import math


class PatchDisNet(nn.Module):  # PatchGAN 用作鉴别器
    def __init__(self, channel, ngf):
        super(PatchDisNet, self).__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        self.conv1 = nn.Sequential(nn.Conv2d(channel, ngf, kernel_size=kw, stride=2, padding=padw),
                                   nn.LeakyReLU(0.2, True))

        self.conv2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=kw, stride=2, padding=padw),
                                   nn.LeakyReLU(0.2, True),
                                   nn.InstanceNorm2d(ngf * 2))

        self.conv3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=kw, stride=2, padding=padw),
                                   nn.LeakyReLU(0.2, True),
                                   nn.InstanceNorm2d(ngf * 4))

        self.conv4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=kw, stride=2, padding=padw),
                                   nn.LeakyReLU(0.2, True),
                                   nn.InstanceNorm2d(ngf * 8))

        self.conv5 = nn.Sequential(nn.Conv2d(ngf * 8, 1, kernel_size=kw, stride=1, padding=padw))

        for m in self.modules():
            ops.weights_init(m)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        return [out1, out2, out3, out4, out5]

class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, ngf, upconv=False, norm=False):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ngf = ngf
        self.norm = norm
        self.upconv = upconv

        if self.norm:
            self.n0 = torch.nn.InstanceNorm2d(self.ngf * 2)
            self.n1 = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n2 = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n3 = torch.nn.InstanceNorm2d(self.ngf * 16)
            self.n3u = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n2u = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n1u = torch.nn.InstanceNorm2d(self.ngf * 2)
        if self.upconv:
            self.u3 = nn.ConvTranspose2d(self.ngf * 16, self.ngf * 16, 3, padding=1, output_padding=1, stride=2)
            self.u2 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 8, 3, padding=1, output_padding=1, stride=2)
            self.u1 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 4, 3, padding=1, output_padding=1, stride=2)
            self.u0 = nn.ConvTranspose2d(self.ngf * 2, self.ngf * 2, 3, padding=1, output_padding=1, stride=2)

        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * 24, self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * 12, self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * 6, self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

        for m in self.modules():
            ops.weights_init(m)

    def forward(self, input_data, inter_mode='nearest'):
        x0 = self.l0(input_data)
        if self.norm:
            x0 = self.n0(x0)
        x1 = self.l1(x0)
        if self.norm:
            x1 = self.n1(x1)
        x2 = self.l2(x1)
        if self.norm:
            x2 = self.n2(x2)
        x3 = self.l3(x2)
        if self.norm:
            x3 = self.n3(x3)
        x3 = self.block1(x3) + x3
        x3 = self.block2(x3) + x3
        if self.upconv:
            x3u = nn.functional.interpolate(self.u3(x3), size=x2.shape[2:4], mode=inter_mode)
        else:
            x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)
        x3u = self.l3u(torch.cat((x3u, x2), dim=1))
        if self.norm:
            x3u = self.n3u(x3u)

        if self.upconv:
            x2u = nn.functional.interpolate(self.u2(x3u), size=x1.shape[2:4], mode=inter_mode)
        else:
            x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        if self.norm:
            x2u = self.n2u(x2u)

        if self.upconv:
            x1u = nn.functional.interpolate(self.u1(x2u), size=x0.shape[2:4], mode=inter_mode)
        else:
            x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(torch.cat((x1u, x0), dim=1))
        if self.norm:
            x1u = self.n1u(x1u)

        if self.upconv:
            x0u = nn.functional.interpolate(self.u0(x1u), size=input_data.shape[2:4], mode=inter_mode)
        else:
            x0u = nn.functional.interpolate(x1u, size=input_data.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        return x0u

##连两层
class SUNet(nn.Module):
    def __init__(self, in_channel, out_channel, ngf, upconv=False, norm=False):
        super(SUNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ngf = ngf
        self.norm = norm
        self.upconv = upconv

        if self.norm:
            self.n0 = torch.nn.InstanceNorm2d(self.ngf * 2)
            self.n1 = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n2 = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n3 = torch.nn.InstanceNorm2d(self.ngf * 16)
            self.n3u = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n2u = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n1u = torch.nn.InstanceNorm2d(self.ngf * 2)
        if self.upconv:
            self.u3 = nn.ConvTranspose2d(self.ngf * 16, self.ngf * 16, 3, padding=1, output_padding=1, stride=2)
            self.u2 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 8, 3, padding=1, output_padding=1, stride=2)
            self.u1 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 4, 3, padding=1, output_padding=1, stride=2)
            self.u0 = nn.ConvTranspose2d(self.ngf * 2, self.ngf * 2, 3, padding=1, output_padding=1, stride=2)

        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * 24, self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * 12, self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

        for m in self.modules():
            ops.weights_init(m)

    def forward(self, input_data, inter_mode='bilinear'):
        x0 = self.l0(input_data)
        if self.norm:
            x0 = self.n0(x0)
        x1 = self.l1(x0)
        if self.norm:
            x1 = self.n1(x1)
        x2 = self.l2(x1)

        if self.norm:
            x2 = self.n2(x2)
        x3 = self.l3(x2)
        if self.norm:
            x3 = self.n3(x3)

        x3 = self.block1(x3) + x3

        x3 = self.block2(x3) + x3

        if self.upconv:
            x3u = nn.functional.interpolate(self.u3(x3), size=x2.shape[2:4], mode=inter_mode)
        else:
            x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)

        x3u = self.l3u(torch.cat((x3u, x2), dim=1))

        if self.norm:
            x3u = self.n3u(x3u)

        if self.upconv:
            x2u = nn.functional.interpolate(self.u2(x3u), size=x1.shape[2:4], mode=inter_mode)
        else:
            x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        if self.norm:
            x2u = self.n2u(x2u)

        if self.upconv:
            x1u = nn.functional.interpolate(self.u1(x2u), size=x0.shape[2:4], mode=inter_mode)
        else:
            x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(x1u)
        if self.norm:
            x1u = self.n1u(x1u)

        if self.upconv:
            x0u = nn.functional.interpolate(self.u0(x1u), size=input_data.shape[2:4], mode=inter_mode)
        else:
            x0u = nn.functional.interpolate(x1u, size=input_data.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        return x0u

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class Attention3(nn.Module):
    def __init__(self, in_channel1, out_channel, ngf, num_heads, norm):
        super(Attention3, self).__init__()
        self.in_channel1 = in_channel1
        self.out_channel = out_channel
        self.ngf = ngf
        self.num_heads = num_heads
        self.norm = norm

        if self.norm:
            self.n0 = torch.nn.InstanceNorm2d(self.ngf * 2)
            self.n1 = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n2 = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n3 = torch.nn.InstanceNorm2d(self.ngf * 16)
            # self.n3u = torch.nn.InstanceNorm2d(self.ngf * 8)
            # self.n2u = torch.nn.InstanceNorm2d(self.ngf * 4)
            # self.n1u = torch.nn.InstanceNorm2d(self.ngf * 2)
            self.n3u = torch.nn.InstanceNorm2d(self.ngf * 16)
            self.n2u = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n1u = torch.nn.InstanceNorm2d(self.ngf * 4)

        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel1, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU()
        )
        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.crossAttention1 = CrossAttention(self.ngf * 8, num_heads=8, attn_drop=0.1, proj_drop=0.1)

        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * (16 + 8), self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * (8 + 4), self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

        for m in self.modules():
            ops.weights_init(m)

    def forward(self, input_data1, inter_mode='bilinear'):  #inter_mode='nearest'
        x0 = self.l0(input_data1)
        if self.norm:
            x0 = self.n0(x0)
        x1 = self.l1(x0)
        if self.norm:
            x1 = self.n1(x1)
        x2 = self.l2(x1)
        if self.norm:
            x2 = self.n2(x2)
        x3 = self.l3(x2)
        if self.norm:
            x3 = self.n3(x3)
        ##
        x3 = self.block1(x3) + x3
        x3 = self.block2(x3) + x3
        ##
        x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)
        #利用高层低噪声特征对低层噪声进行注意力降噪
        x3_a = self.crossAttention1(x3u,x2)
        #x3u = self.l3u(torch.cat((x3u, x2), dim=1))
        x3u = self.l3u(x3_a)
        if self.norm:
            x3u = self.n3u(x3u)
        x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        if self.norm:
            x2u = self.n2u(x2u)
        x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(x1u)
        if self.norm:
            x1u = self.n1u(x1u)
        x0u = nn.functional.interpolate(x1u, size=input_data1.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        return x0u

class AttentionNet(nn.Module):
    def __init__(self, in_channel1, out_channel, ngf, num_heads, norm):
        super(AttentionNet, self).__init__()
        self.in_channel1 = in_channel1
        self.out_channel = out_channel
        self.ngf = ngf
        self.num_heads = num_heads
        self.norm = norm

        if self.norm:
            self.n0 = torch.nn.InstanceNorm2d(self.ngf * 2)
            self.n1 = torch.nn.InstanceNorm2d(self.ngf * 4)
            self.n2 = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n3 = torch.nn.InstanceNorm2d(self.ngf * 16)
            # self.n3u = torch.nn.InstanceNorm2d(self.ngf * 8)
            # self.n2u = torch.nn.InstanceNorm2d(self.ngf * 4)
            # self.n1u = torch.nn.InstanceNorm2d(self.ngf * 2)
            self.n3u = torch.nn.InstanceNorm2d(self.ngf * 16)
            self.n2u = torch.nn.InstanceNorm2d(self.ngf * 8)
            self.n1u = torch.nn.InstanceNorm2d(self.ngf * 4)

        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel1, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU()
        )
        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU()
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )
        self.crossAttention1 = CrossAttention(self.ngf * 8, num_heads=8, attn_drop=0.1, proj_drop=0.1)
        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * (16 + 8), self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * (8 + 4), self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * (4 + 2), self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU()
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

        for m in self.modules():
            ops.weights_init(m)

    def forward(self, input_data1, inter_mode='bilinear'):  #inter_mode='nearest'
        x0 = self.l0(input_data1)
        if self.norm:
            x0 = self.n0(x0)
        x1 = self.l1(x0)
        if self.norm:
            x1 = self.n1(x1)
        x2 = self.l2(x1)
        if self.norm:
            x2 = self.n2(x2)
        x3 = self.l3(x2)
        if self.norm:
            x3 = self.n3(x3)
        ##
        x3 = self.block1(x3) + x3
        x3 = self.block2(x3) + x3
        ##
        x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)
        #利用高层低噪声特征对低层噪声进行注意力降噪
        x3_a = self.crossAttention1(x3u,x2)
        #x3u = self.l3u(torch.cat((x3u, x2), dim=1))
        x3u = self.l3u(x3_a)
        if self.norm:
            x3u = self.n3u(x3u)
        x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        if self.norm:
            x2u = self.n2u(x2u)
        x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(torch.cat((x1u, x0), dim=1))
        if self.norm:
            x1u = self.n1u(x1u)
        x0u = nn.functional.interpolate(x1u, size=input_data1.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        return x0u

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(2 * dim, dim, bias=qkv_bias)
        self.k = nn.Linear(2 * dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        B1, C1, H1, W1 = x.shape
        x = x.reshape(B1, C1, H1 * W1)
        x = x.permute(0, 2, 1)
        ##
        B2, C2, H2, W2 = y.shape
        y = y.reshape(B2, C2, H2 * W2)
        y = y.permute(0, 2, 1)
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, -1, self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B2, -1, self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3)
        attn1 = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn1.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.v(y).reshape(B2, -1, self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3)
        x_att = (attn @ v).transpose(1, 2).reshape(B1, -1, C2)
        x_att = self.proj(x_att)
        x_att = self.proj_drop(x_att)
        ##
        x = x.permute(0, 2, 1)
        x = x.reshape(B1, C1, H1, W1)
        x_att = x_att.permute(0, 2, 1)
        x_att = x_att.reshape(B1, C2, H1, W1)
        x1 = torch.cat((x, x_att), dim=1)
        ##
        return x1


if __name__ == '__main__':
    x = torch.randn(4, 4, 424, 424).cuda()
    y = torch.randn(1, 512, 27, 27).cuda()

