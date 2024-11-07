import torch.nn as nn
from .mlp import MLP
from utils.utils import input_mapping, grid_coordinate

class INR(nn.Module):
#用于拟合的隐式神经表达
    def __init__(self, args):
        super().__init__()
        in_dim = args.input_mappings * 4 + 2
        # input_mapping
        self.L = args.input_mappings
        self.inr_net = MLP(in_dim, args.output_channel, args.hidden_list, args.activation_function)

    def forward(self, x):
        """
        :param x: 输入场 B * C * H * W
        :return: 经过拟合得到的输出场 B * C * H * W
        """
        shape = x.shape[-2:]  # H * W
        coord = grid_coordinate(shape).to(x.device)  # HW * 2
        coord_encode = input_mapping(coord, self.L)
        output = self.inr_net(coord_encode)
        return  output.clone()