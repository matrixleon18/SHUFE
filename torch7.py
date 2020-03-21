# encoding=GBK
"""
创建神经网络并绘图
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

NUM_TOTAL_POINTS = 40

n_data = torch.ones(int(NUM_TOTAL_POINTS/2), 2)                                     # 创建一群100个点，数据基准坐标(1,1)

x0 = torch.normal(2*n_data, 1)                                                      # 对点的坐标进行以均值为２方差为１的分布
y0 = torch.zeros(int(NUM_TOTAL_POINTS/2))                                           # 这一半的点分类标记为０

x1 = torch.normal(-2*n_data, 1)                                                     # 对点坐标进行以均值为－２方差为１的分布
y1 = torch.ones(int(NUM_TOTAL_POINTS/2))                                             # 这一半的点标记为１

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)                              # 把x0和x1按照维度１进行拼接（竖着）
y = torch.cat((y0, y1), ).type(torch.LongTensor)                                # 把y0和y1进行拼接

x, y = Variable(x), Variable(y)                                                 # 把x,y都转化成Variable


# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(2, 10, 2)                                         # 输入两个值[横坐标,纵坐标]；输出onehot编码的两个值[0,1]或[1,0]表示两类
# print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()                     # CrossEntropyLoss/交叉熵损失:就是针对分类问题的损失函数。特别是多分类问题。计算的是概率

for t in range(100):
        out = net(x)                                        # 输出的是[200,2]的tensor,因为是200个点,每个点计算出来的值类似[-3,4],
        loss = loss_func(out, y)                            # 计算交叉熵,类似tensor(0.3932, grad_fn=<NllLossBackward>)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 2 == 0:
            plt.cla()
            out_softmax = torch.softmax(out, -2)            # 将输出归一化到(0,1)中，还是[20,2]的tensor，是概率,列和为１
            ret = torch.max(out_softmax, 1)                 # 取出每一行的最大值和其索引,组成tensor(valuns=[20], indics=[20])
            prediction = ret[1]                             # 得到每一行最大值的索引 [20]
            pred_y = prediction.data.numpy().squeeze()      # 转换成numpy数据，再转成一维array[20]，这就是预测的ｙ分类[1,0,0,...]
            target_y = y.data.numpy()                       # 这是实际的ｙ分类
            plt.scatter(x.data.numpy()[:, 0],               # 输入点的X坐标
                        x.data.numpy()[:, 1],               # 输入点的y坐标
                        c=pred_y,                           # 这是点的颜色。不同的数字代表了不同的颜色。
                        s=100,                              # 点的大小
                        lw=0)                               # 线宽
            accuracy = sum(pred_y == target_y) / NUM_TOTAL_POINTS   # 计算预测正确的概率
            plt.text(0, -3, "Round: {} Accuracy: {}".format(t, accuracy))
            plt.pause(0.1)


plt.ioff()
plt.show()