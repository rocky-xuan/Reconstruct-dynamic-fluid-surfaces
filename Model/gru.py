import torch
import torch.nn as nn

# 定义GRU模型
class DecoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)  # 使用GRU层

        # 输出层
        self.fc = nn.Linear(hidden_size, input_size)  # 输出大小为512

    def forward(self, y):
        # 初始化隐藏状态
        h0 = torch.zeros(1, y.size(0), self.hidden_size).to(y.device)

        # 前向传播
        out, _ = self.gru(y, h0)

        # 输出层
        x = self.fc(out)

        # 为了确保输出在[0, 1]范围内，可以使用sigmoid函数
        x = torch.sigmoid(x)

        return x

if __name__ == '__main__':

    # 创建模型
    input_size = 512
    hidden_size = 512  # 调整隐藏层大小为512，以匹配输入和输出的大小
    decoder_gru = DecoderGRU(input_size, hidden_size)
    input = torch.rand(1, 512, 512)
    output = decoder_gru(input)
    print(output.shape)
