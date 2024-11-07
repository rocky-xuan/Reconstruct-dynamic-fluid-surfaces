import torch.nn as nn
from .mlp import MLP
from utils.utils import input_mapping, grid_coordinate
import torch
class INR(nn.Module):
#用于拟合的隐式神经表达
    def __init__(self, args):
        super().__init__()
        in_dim = args.input_mappings * 4 + 2
        # input_mapping
        self.L = args.input_mappings
        self.inr_net = MLP(in_dim, args.output_channel, args.hidden_list, args.activation_function)

    def forward(self, x, bias):
        """
        :param x: 输入场 B * C * H * W
        :return: 经过拟合得到的输出场 B * C * H * W
        """
        shape = x.shape[-2:]  # H * W
        grids = []
        # coord = grid_coordinate(shape)
        for index, item in enumerate(shape):
            min, max = -1, 1  # -1~1有利于后续的傅里叶编码
            r = (max - min) / (2 * item)
            grid = min + r + (2 * r) * torch.arange(item).float()
            grid = grid + r / 100 * bias
            mins = torch.ones_like(grid) * min
            maxs = torch.ones_like(grid) * max
            torch.where(grid < -1, mins, grid)
            torch.where(grid > 1, maxs, grid)
            grids.append(grid)
        ret = torch.stack(torch.meshgrid(*grids), dim=-1)
        ret = ret.view(-1, ret.shape[-1])
        coord = ret.to(x.device)  # HW * 2

        coord_encode = input_mapping(coord, self.L)
        output = self.inr_net(coord_encode)
        return  output.clone()