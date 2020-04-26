# encoding=GBK

"""
ѭ��������ع�
"""

import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)                        # ���������

TIME_STEP = 10                              # rnn ʱ�䲽��
INPUT_SIZE = 1                              # ��Ϊ�����֣�����1��feature
LR = 0.02                                   # learning rate
HIDDEN_SIZE = 64                            # ���ز����������


class RNN(nn.Module):                                   # hidden = np.tanh(np.dot(self.W_hh, hidden) + np.dot(self.W_xh, x))
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(                              # �Լ�����һ��RNN������
            input_size=INPUT_SIZE,                      # ���� x ��feature ά��
            hidden_size=HIDDEN_SIZE,                    # ��״̬ hidden_state �е�featureά��
            num_layers=1,                               # RNN �Ĳ���
            nonlinearity='tanh',                        # ָ������� [��tanh�� | ��relu��]. Ĭ��: ��tanh��
            bias=True,                                  # ����� False , ��ô RNN ��Ͳ���ʹ��ƫ��Ȩ�� b_ih �� b_hh, Ĭ��: True
            batch_first=True,                           # ��� True, ����Tensor��shapeӦ����(batch, seq, features),�������Ҳ��һ��.
            dropout=0,                                  # ���ֵ����, ��ô�������һ����, ������������������һ�� dropout ��
            bidirectional=False                         # ��� True , ������һ��˫�� RNN, Ĭ��Ϊ False
        )
        self.out = nn.Linear(HIDDEN_SIZE, 1)            # ����һ������㣬����RNN����������ֻ�����output_vector

    def forward(self, x, h_state):                      # �����RNNÿ������Ĳ���x��h
        # x (batch, time_step, input_size)              # ����RNN�ģ���ά��            (����, ���г���, ���������ά�ȣ�
        # h_state (n_layers, batch, hidden_size)        # ����hidden_state��ά��       (����������, ����, ���������ά�ȣ�/*���򣺵����ǣ���˫���ǣ�*/
        # r_out (batch, time_step, hidden_size)         # ��������ʵ�������r_out��ά�� (���������г��ȣ����������ά��X����
        r_out, h_state = self.rnn(x, h_state)           # RNNÿ������x, hidden_state; ���r_out, hidden_state;
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))

        return torch.stack(outs, dim=1), h_state        # RNN��forward�����output_vector, hidden_state


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None                                                          # ��ʼʱ�����RNN��hidden_state����None

for step in range(200):                                                 # ���㣲�����Ρ��൱�ڣ�������˳���ʱ��Ƭ���ݶ���ȥ����
    start, end = step*np.pi, (step+1)*np.pi                             # ���һС��������ʼ��
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)        # ����һС������
    x_np = np.sin(steps)                                                # ������������������
    y_np = np.cos(steps)                                                # �������Ҫ��Ԥ�������
    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))     # shape 1D -> 3D
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))
    # print(x.size())
    # print(y.size())
    prediction, h_state = rnn(x, h_state)                               # �����һ��RNNѵ�������Ľ��
    h_state = Variable(h_state.data)                                    # ��tensor�е�����ȡ����

    loss = loss_func(prediction, y)
    print(loss.data.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.ion()                                                           # ��ʵ��ͼ��Ԥ��ͼ��̬��ӡ����
    plt.plot(steps, y_np, color='b')
    plt.plot(steps, np.squeeze(prediction.data.numpy()), color='r')
    plt.show()
    plt.pause(0.30)