import torch.nn as nn
import torch
import torch.nn.functional as F
from .unet_parts import OutConv,DoubleConv,Down
from torchsummary import summary


class SparseSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SparseSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=2)

        value = self.value_conv(x).view(batch_size, -1, height * width)
        sparse_attention = torch.bmm(value, attention.permute(0, 2, 1))
        sparse_attention = sparse_attention.view(batch_size, channels, height, width)

        out = self.gamma * sparse_attention + x
        return out


class UNetWithSparseSelfAttention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetWithSparseSelfAttention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 4)
        self.down1 = Down(4, 8)
        self.down2 = Down(8, 16)
        # self.down3 = Down(16, 32)
        # self.down4 = Down(32, 32)

        # 添加稀疏自注意力机制到解码器层
        # self.up1 = UpWithSparseSelfAttention(64, 16, bilinear)
        # self.up2 = UpWithSparseSelfAttention(32, 8, bilinear)
        self.up3 = UpWithSparseSelfAttention(24, 4, bilinear)
        self.up4 = UpWithSparseSelfAttention(8, 4, bilinear)

        self.outc = OutConv(4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.shape)

        x = self.up1(x5, x4)
        # print(x.shape)
        x = self.up2(x, x3)
        x = self.up3(x3, x2)
        # print(x.shape)

        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


# 定义具有稀疏自注意力机制的Up模块
class UpWithSparseSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpWithSparseSelfAttention, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

        # 添加稀疏自注意力模块
        self.sparse_attention = SparseSelfAttention(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 将x1和x2连接起来
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)

        # 应用稀疏自注意力
        x = self.sparse_attention(x)

        return x
if __name__ == '__main__':
    device = torch.device('cuda')
    net = UNetWithSparseSelfAttention(n_channels=2, n_classes=1).to(device)
    summary(net, input_size=(2, 128, 128))
    print(net)
    t = torch.rand([1,2,128,128]).to(device)
    o = net(t)
    print(o.shape)
