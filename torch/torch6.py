# encoding=GBK

"""
������������϶�������
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)      # �������ƶ�λ�ã��ϼ���ά��Ϊ����ά�ȣ�ת���ɣ������У���
# print(x.shape)
y = x.pow(2) + 0.2*torch.rand(x.size())                     # �ڷ������ɵĵ������������һЩnoise
# plt.scatter(x,y)
# plt.show()

x, y = Variable(x), Variable(y)                             # ������ֻ�ܽ��� Variable
# plt.scatter(x.data.numpy(), y.data.numpy())               # ��ӡ��ԭʼ�ĵ�
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):                      # ��ʼ������
        super(Net, self).__init__()                                         # ���ø���ĳ�ʼ������
        self.hidden = torch.nn.Linear(n_feature, n_hidden)                  # ����һ�����ز�
        self.predict = torch.nn.Linear(n_hidden, n_output)                  # �����
        pass

    def forward(self, x):                       # ǰ�򴫵ݺ���
        # x = torch.relu(self.hidden(x))        # Ч������
        x = torch.tanh(self.hidden(x))          # Ч���ܺá����ǲ�������
        # x = torch.sigmoid(self.hidden(x))     # Ч���ܲ���
        # x = F.softmax(self.hidden(x))         # Ч���ܲ���
        x = self.predict(x)
        return x

net = Net(1, 10, 1)                 # ��������ֵ����������Ԫ�����ز㣬�������ֵ
# print(net)

# show it
plt.ion()                           # pyplot ���������ģʽ
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)                   # �������Ż������Ż�������������Ĳ�����ѧϰ����0.2
loss_func = torch.nn.MSELoss()                                          # ��������ʧ��������������ʧ���� MeanSquareError

for t in range(100):                                                    # ѭ������200��
    prediction = net(x)                                                 # ����������������Ԥ��ֵ
    loss = loss_func(prediction, y)                                     # ����Ԥ��ֵ��ʵ��ֵ֮��Ĳ�ֵ

    optimizer.zero_grad()                                               # �ݶ��Ƚ�Ϊ��
    loss.backward()                                                     # ���򴫵�
    optimizer.step()                                                    # �ݶ��½�

    if t % 5 == 0:
        plt.clf()                                                                               # ��������ᣬ���Ǵ��ڴ򿪣����������Ա��ظ�ʹ�á�
        plt.scatter(x.data.numpy(), y.data.numpy())                                             # ���Ƴ�ԭʼ�����ĵ�
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=3)                           # ���Ƴ�Ԥ��ĵ�
        plt.text(0.5, 0, 'Round:{} Loss :{}'.format(t, round(float(loss.data.numpy()), 5)))     # ��ӡ�����
        plt.pause(0.1)                                                                          # ��ͣ��.����


plt.ioff()              # ����������ģʽ
plt.show()              # ��ʾͼ��
