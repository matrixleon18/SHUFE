# encoding=GBK

"""
卷积神经网络 CNN
"""


import torch
import torch.nn as nn
import torch.utils.data.dataloader as loader
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt


# hyper parameter
EPOCH = 2
BATCH_SIZE = 50
LR = 0.005
DOWNLOAD_MNIST = False                                  # 第一次运行把这里设成True来下载数据集。以后就是False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get train data
train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,                                         # 现在load的是 MNIST 里面的train数据。有60K
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)


# Get test data
test_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=False
)


# show the test data
# print(train_data.data.size())                           # 这是导入的 MNIST 训练数据大小 [60000, 28, 28]格式
# print(train_data.targets.size())                        # 这是导入的 MNIST 数据对应的真实数字 [60000]格式
# print(train_data.data[0])                               # 目前的图片数据还是属于(0,255)区间内
# plt.imshow(train_data.data[0].numpy())                  # 把 MNIST 第一个图像打出来看看
# print(train_data.targets[0])                            # 是 MNITST 第一个图像标注出来的数字。也就是我们预测的目标
# plt.show()

train_loader = loader.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

with torch.no_grad():                                     # 下面的数据不更新计算梯度，因为都是自己定义的正向计算，不用记录计算图图来反向传播，不作预测
    test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:2000]/255    # 数据/255为了压缩在(0,1)区间内
    test_y = test_data.targets[:2000]                                                               # 先试验2000个数据


# print(test_x.size())                                        # 现在数据变成了[2000, 1, 28, 28]格式
# print(test_x[0][0])                                         # 现在图片数据被压缩在了(0,1)区间内
# plt.imshow(test_x[0][0].numpy())                            # 不过图片还是显示没有变化
# plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(                                      # 2D卷积层. 输入的数据size = channel*W*H = 1*28*28
                in_channels=1,                              # 输入的维度
                out_channels=16,                            # 输出的filter数量，也就是16层的图
                kernel_size=5,                              # 采样窗口的大小是5x5像素
                stride=1,                                   # 每次移动的步长
                padding=2,                                  # 图像边界补的像素数量，因为(kernel_size-1)/2=2
            ),                                              # 输出数据 size = 16*28*28
            nn.ReLU(),                                      # 卷积层. 输出同上
            nn.MaxPool2d(
                kernel_size=2,                              # 在2x2区域内选取最大的值
            )                                               # 输出数据size=16*14*14 （因为２取１）
        )

        self.conv2 = nn.Sequential(                         # 输入数据size=16*14*14
            nn.Conv2d(16, 32, 5, 1, 2),                     # 参考上面的 Conv2d 参数; 输出数据size=32*14*14
            nn.ReLU(),                                      # 输出数据size=32*14*14
            nn.MaxPool2d(2)                                 # 输出数据size=32*7*7
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # x.to(device)
        x = self.conv1(x)
        x = self.conv2(x)                           # (batch, 32, 7, 7)
        output = x.view(x.size(0), -1)                   # (batch, 32*7*7)
        output = self.out(output)
        return output


cnn = CNN()                                                 # 生成一个卷积网络实例
# print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)       # 优化这个卷积神经网络里所有的参数
loss_func = nn.CrossEntropyLoss()                           # 损失函数

for epoch in range(EPOCH):                                  # 开始训练了
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = int(sum(pred_y == test_y)) / test_y.size(0)
            print("Epoch: {}, loss: {}, accuracy: {}".format(epoch, round(float(loss.data), 2), accuracy))


test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print("pred: ", pred_y)
print("real: ", test_y[:10].numpy())