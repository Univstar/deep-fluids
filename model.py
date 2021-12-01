from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from legacy.ops import upscale


class Network(nn.Module):
    def __init__(self, shape):
        super(Network, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, shape.prod(dtype=torch.int).item()),
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output


class ResModel(nn.Module):
    def __init__(self, conv_num, conv_args, scale=None):
        super(ResModel, self).__init__()
        self.conv = nn.Sequential()
        for i in range(conv_num):
            self.conv.add_module(f'conv{i}', nn.Conv2d(**conv_args))
            self.scale = scale
        self.scale = scale
        if self.scale is not None:
            self.scale = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x0):
        x = self.conv(x0) + x0
        if self.scale is not None:
            x = self.scale(x)
        return x


class DFModel(nn.Module):
    def __init__(self, input_shape, channel_num, conv_num, output_shape):
        super(DFModel, self).__init__()
        layer_num = int(np.log2(np.max(output_shape[:-1]))) - 2

        self.first_shape = [int(i/np.power(2, layer_num - 1))
                       for i in output_shape[:-1]]

        self.fc = nn.Linear(input_shape, np.prod(self.first_shape))

        self.conv = nn.Sequential()

        conv_args = {
            'in_channels': channel_num,
            'out_channels': channel_num,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        }

        self.conv.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=channel_num, kernel_size=3, stride=1, padding=1))
        for i in range(layer_num):
            if i < layer_num-1:
                self.conv.add_module(f'res{i}', ResModel(conv_num, conv_args, 2))
            else:
                self.conv.add_module(f'res{i}', ResModel(conv_num, conv_args, None))
                self.conv.add_module(f'conv{i}', nn.Conv2d(
                    in_channels=channel_num, out_channels=output_shape[-1], kernel_size=3, stride=1, padding=1))

    def curl(self, x):
        u = x[..., 1:, :] - x[..., :-1, :]  # ds/dy
        v = x[..., :, :-1] - x[..., :, 1:]  # -ds/dx,
        u = torch.cat([u, torch.unsqueeze(u[..., -1, :], dim=-2)], dim=-2)
        v = torch.cat([v, torch.unsqueeze(v[..., :, -1], dim=-1)], dim=-1)
        c = torch.cat([u, v], dim=-1)
        return c

    def forward(self, x0):
        first_layer = self.fc(x0)
        
        output = self.conv(first_layer.view(-1,1, *self.first_shape))
        return self.curl(output)
