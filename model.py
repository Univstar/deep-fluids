import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResModel(nn.Module):
    def __init__(self, num_conv, conv_args, factor=1):
        super(ResModel, self).__init__()
        self.conv = nn.Sequential()
        for i in range(num_conv):
            self.conv.add_module(f'conv{i}', nn.Conv2d(**conv_args))
            self.conv.add_module(f'relu{i}', nn.LeakyReLU(0.2))
        self.scale = None
        if factor != 1:
            self.scale = nn.UpsamplingNearest2d(scale_factor=factor)

    def forward(self, x0):
        x = self.conv(x0) + x0
        if self.scale is not None:
            x = self.scale(x)
        return x


class DFModel(nn.Module):
    def __init__(self, input_shape, num_chnl, num_conv, output_shape):
        super(DFModel, self).__init__()
        layer_num = int(np.log2(np.max(output_shape[:-1]))) - 2

        self.first_shape = [num_chnl] + [int(i/np.power(2, layer_num - 1)) for i in output_shape[:-1]]

        self.fc = nn.Linear(input_shape, np.prod(self.first_shape))

        self.conv = nn.Sequential()

        conv_args = {
            'in_channels': num_chnl,
            'out_channels': num_chnl,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        }

        for i in range(layer_num):
            if i < layer_num-1:
                self.conv.add_module(f'res{i}', ResModel(num_conv, conv_args, 2))
            else:
                self.conv.add_module(f'res{i}', ResModel(num_conv, conv_args))
        self.conv.add_module(f'conve', nn.Conv2d(in_channels=num_chnl, out_channels=output_shape[-1], kernel_size=3, stride=1, padding=1))

    def forward(self, x0):
        first_layer = self.fc(x0)
        output = self.conv(first_layer.view(-1, *self.first_shape))
        return output
