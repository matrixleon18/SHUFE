# encoding=GBK
"""
神经网络的创建，存储，提取
"""


import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


torch.manual_seed(1)                                            # 为CPU设置种子用于生成随机数，以使得得到随机数是确定的

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)          # 生成了一个[100, 1]的数组；值是从－１到１的１００等差数列
y = x.pow(2) + 0.2*torch.rand(x.size())                         # 把ｘ平方一下，再加上一个很小的随机数作为Noise
x, y = Variable(x), Variable(y)                                 # 转换成 Variable

def save_net():
    net1 = torch.nn.Sequential(                                 # 定义一个简易神经网络
        torch.nn.Linear(1, 100),                                # 输入节点数inputSize=1，输出节点数outputSize=100.init会生成
                                                                # self.weight=torch.Tensor(outputSize, inputSize)
                                                                # self.bias = torch.Tensor(outputSize)
                                                                # self.gradWeight = torch.Tensor(outputSize, inputSize)
                                                                # self.gradBias = torch.Tensor(outputSize)
                                                                # parent.__init() 生成 gradInput, output, _type
                                                                # 所以Liner一共是７个参数
        torch.nn.ReLU(inplace=True),                            # 激活层. inplace=True对计算结果不会有影响。
                                                                # 利用in-place计算可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。但是会对原变量覆盖，只要不带来错误就用
        torch.nn.Linear(100, 1)                                 # 输入１００，　输出１
    )

    # print(list(net1.parameters()))

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.01)     # 第一个是需要被优化的参数，其形式必须是Tensor或者dict；第二个是优化选项，包括学习率、衰减率等
                                                                # 定义SGD优化器;optimizer存了net1所有parameters的指针;
                                                                # SGD继承了Optimizer, 在Optimizer类的__init__()里,
                                                                # 通过add_param_group()添加进了self.param_groups 属性params的字典里。
                                                                # 每个parameter都包含梯度（gradient）optimizer可以把梯度应用上去更新parameter

    loss_func = torch.nn.MSELoss()                              # 定义损失函数.
                                                                # 对prediction和y之间进行比对（熵或者其他loss function），产生最初的梯度

    for epoch in range(1000):                                   # 开始1000个epoch进行训练
        prediction = net1(x)                                    # 把观测点输入模型中得到预测值
        loss = loss_func(prediction, y)                         # 通过损失函数计算出观测值和预测值的误差
        optimizer.zero_grad()                                   # 清空优化器上次留下来的参数梯度信息

        loss.backward()                                         # 把误差进行网络反向传递回去
                                                                # 反向传播到整个网络的所有链路和节点。节点与节点之间有联系，因此可以反向链式传播梯度。

        optimizer.step()                                        # 优化器就可以根据loss来更新网络参数的梯度。参照SGD代码
                                                                # 根据gradient更新网络参数，d.grad是loss.backward()自动求导机制已计算出了参数的gradient
                                                                # 根据这些parameter的gradient对parameter的值进行更新。这里不需要传入梯度，因为梯度的指针已经在其构造函数中传入的parameters中包含

    torch.save(net1, "net1.pkl")
    torch.save(net1.state_dict(), "net1_param.pkl")

    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title("Net1")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)

def restore_net():
    net2 = torch.load("net1.pkl")

    prediction = net2(x)

    plt.figure(1, figsize=(10, 3))
    plt.subplot(132)
    plt.title("Net2")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)


def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1)
    )
    net3.load_state_dict(torch.load("net1_param.pkl"))

    prediction = net3(x)

    plt.figure(1, figsize=(10, 3))
    plt.subplot(133)
    plt.title("Net3")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)


save_net()
restore_net()
restore_params()
plt.show()
