# encoding=GBK

"""
循环神经网络回归
"""

import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)                        # 随机数种子

TIME_STEP = 10                              # rnn 时间步数
INPUT_SIZE = 1                              # 因为是数字，就是1个feature
LR = 0.02                                   # learning rate
HIDDEN_SIZE = 64                            # 隐藏层的特征数量


class RNN(nn.Module):                                   # hidden = np.tanh(np.dot(self.W_hh, hidden) + np.dot(self.W_xh, x))
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(                              # 自己定义一个RNN神经网络
            input_size=INPUT_SIZE,                      # 输入 x 的feature 维度
            hidden_size=HIDDEN_SIZE,                    # 隐状态 hidden_state 中的feature维度
            num_layers=1,                               # RNN 的层数
            nonlinearity='tanh',                        # 指定激活函数 [‘tanh’ | ’relu’]. 默认: ‘tanh’
            bias=True,                                  # 如果是 False , 那么 RNN 层就不会使用偏置权重 b_ih 和 b_hh, 默认: True
            batch_first=True,                           # 如果 True, 输入Tensor的shape应该是(batch, seq, features),并且输出也是一样.
            dropout=0,                                  # 如果值非零, 那么除了最后一层外, 其它层的输出都会套上一个 dropout 层
            bidirectional=False                         # 如果 True , 将会变成一个双向 RNN, 默认为 False
        )
        self.out = nn.Linear(HIDDEN_SIZE, 1)            # 定义一个输出层，这是RNN的最后输出，只用输出output_vector

    def forward(self, x, h_state):                      # 这就是RNN每次输入的参数x和h
        # x (batch, time_step, input_size)              # 这是RNN的ｘ的维度            (批量, 序列长度, 输入的特征维度）
        # h_state (n_layers, batch, hidden_size)        # 这是hidden_state的维度       (层数×方向, 批量, 输出的特征维度）/*方向：单向是１；双向是２*/
        # r_out (batch, time_step, hidden_size)         # 这是网络实际输出的r_out的维度 (批量，序列长度，输出的特征维度X方向）
        r_out, h_state = self.rnn(x, h_state)           # RNN每次输入x, hidden_state; 输出r_out, hidden_state;
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))

        return torch.stack(outs, dim=1), h_state        # RNN的forward输出了output_vector, hidden_state


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None                                                          # 起始时输入给RNN的hidden_state就是None

for step in range(200):                                                 # 计算２００次。相当于２００个顺序的时间片数据丢进去计算
    start, end = step*np.pi, (step+1)*np.pi                             # 设计一小段数据起始点
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)        # 生成一小段数据
    x_np = np.sin(steps)                                                # 这就是用来输入的数据
    y_np = np.cos(steps)                                                # 这就是需要被预测的数据
    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))     # shape 1D -> 3D
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))
    # print(x.size())
    # print(y.size())
    prediction, h_state = rnn(x, h_state)                               # 这就是一次RNN训练出来的结果
    h_state = Variable(h_state.data)                                    # 把tensor中的数据取出来

    loss = loss_func(prediction, y)
    print(loss.data.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.ion()                                                           # 把实际图和预测图动态打印出来
    plt.plot(steps, y_np, color='b')
    plt.plot(steps, np.squeeze(prediction.data.numpy()), color='r')
    plt.show()
    plt.pause(0.30)