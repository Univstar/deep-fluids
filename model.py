import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResModel(nn.Module):
    def __init__(self, num_conv, conv_args, scale=None):
        super(ResModel, self).__init__()
        self.conv = nn.Sequential()
        for i in range(num_conv):
            self.conv.add_module(f'conv{i}', nn.Conv2d(**conv_args))
            self.conv.add_module(f'relu{i}', nn.LeakyReLU(0.2))
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
    def __init__(self, input_shape, num_chnl, num_conv, output_shape):
        super(DFModel, self).__init__()
        layer_num = int(np.log2(np.max(output_shape[:-1]))) - 2

        self.first_shape = [num_chnl] + [int(i/np.power(2, layer_num - 1))
                       for i in output_shape[:-1]]

        self.fc = nn.Sequential(
            nn.Linear(input_shape, np.prod(self.first_shape)),
            nn.LeakyReLU(0.2))

        self.conv = nn.Sequential()

        conv_args = {
            'in_channels': num_chnl,
            'out_channels': num_chnl,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        }

        self.conv.add_module('convb', nn.Conv2d(in_channels=num_chnl, out_channels=num_chnl, kernel_size=3, stride=1, padding=1))
        self.conv.add_module('relub', nn.LeakyReLU(0.2))
        for i in range(layer_num):
            if i < layer_num-1:
                self.conv.add_module(f'res{i}', ResModel(num_conv, conv_args, 2))
            else:
                self.conv.add_module(f'res{i}', ResModel(num_conv, conv_args, None))
        self.conv.add_module(f'conve', nn.Conv2d(in_channels=num_chnl, out_channels=output_shape[-1], kernel_size=3, stride=1, padding=1))
        self.conv.add_module(f'relue', nn.LeakyReLU(0.2))

    def curl(self, x):
        u = x[..., 1:, :] - x[..., :-1, :]  # ds/dy
        v = x[..., :, :-1] - x[..., :, 1:]  # -ds/dx,
        u = torch.cat([u, torch.unsqueeze(u[..., -1, :], dim=-2)], dim=-2)
        v = torch.cat([v, torch.unsqueeze(v[..., :, -1], dim=-1)], dim=-1)
        c = torch.cat([u, v], dim=-1)
        return c

    def forward(self, x0):
        first_layer = self.fc(x0)
        
        output = self.conv(first_layer.view(-1, *self.first_shape))
        return self.curl(output)
