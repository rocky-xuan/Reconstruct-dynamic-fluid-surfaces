import torch
import torch.nn as nn

# 定义UNet模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super(UNet, self).__init__()

        # 编码器路径
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 解码器路径
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # 编码器路径
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)

        # 解码器路径
        for decoder_layer in self.decoder:
            x = decoder_layer(x)

        return x

# 创建UNet模型实例
in_channels = 3  # 输入通道数
out_channels = 1  # 输出通道数
dropout_prob = 0.5  # Dropout丢弃概率
model = UNet(in_channels, out_channels, dropout_prob)

# 打印模型结构
print(model)
