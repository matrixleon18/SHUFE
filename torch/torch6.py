# encoding=GBK

"""
用神经网络来拟合二次曲线
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)      # 数据在制定位置１上加上维度为１的维度，转化成１００行１列
# print(x.shape)
y = x.pow(2) + 0.2*torch.rand(x.size())                     # 在方程生成的点ｙ坐标后面加上一些noise
# plt.scatter(x,y)
# plt.show()

x, y = Variable(x), Variable(y)                             # 神经网络只能接受 Variable
# plt.scatter(x.data.numpy(), y.data.numpy())               # 打印出原始的点
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):                      # 初始化函数
        super(Net, self).__init__()                                         # 调用父类的初始化函数
        self.hidden = torch.nn.Linear(n_feature, n_hidden)                  # 创建一个隐藏层
        self.predict = torch.nn.Linear(n_hidden, n_output)                  # 输出层
        pass

    def forward(self, x):                       # 前向传递函数
        # x = torch.relu(self.hidden(x))        # 效果还行
        x = torch.tanh(self.hidden(x))          # 效果很好。但是波动蛮大
        # x = torch.sigmoid(self.hidden(x))     # 效果很不好
        # x = F.softmax(self.hidden(x))         # 效果很不好
        x = self.predict(x)
        return x

net = Net(1, 10, 1)                 # １个输入值，１０个神经元的隐藏层，１个输出值
# print(net)

# show it
plt.ion()                           # pyplot 进入非阻塞模式
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)                   # 定义了优化器：优化对象是神经网络的参数，学习率是0.2
loss_func = torch.nn.MSELoss()                                          # 定义了损失函数：均方差损失函数 MeanSquareError

for t in range(100):                                                    # 循环计算200次
    prediction = net(x)                                                 # 根据神经网络计算出了预测值
    loss = loss_func(prediction, y)                                     # 计算预测值和实际值之间的差值

    optimizer.zero_grad()                                               # 梯度先降为０
    loss.backward()                                                     # 反向传递
    optimizer.step()                                                    # 梯度下降

    if t % 5 == 0:
        plt.clf()                                                                               # 清除所有轴，但是窗口打开，这样它可以被重复使用。
        plt.scatter(x.data.numpy(), y.data.numpy())                                             # 绘制出原始给出的点
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=3)                           # 绘制出预测的点
        plt.text(0.5, 0, 'Round:{} Loss :{}'.format(t, round(float(loss.data.numpy()), 5)))     # 打印出误差
        plt.pause(0.1)                                                                          # 暂停０.１秒


plt.ioff()              # 结束非阻塞模式
plt.show()              # 显示图像
