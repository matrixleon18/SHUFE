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
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False                                  # 第一次运行把这里设成True来下载数据集。以后就是False

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

test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:2000]/255    # 数据/255为了压缩在(0,1)区间内
test_y = test_data.targets[:2000]

print(test_x.size())                                        # 现在数据变成了[2000, 1, 28, 28]格式
print(test_x[0][0])                                         # 现在图片数据被压缩在了(0,1)区间内
plt.imshow(test_x[0][0].numpy())                            # 不过图片还是显示没有变化
plt.show()
# with torch.no_grad:                                     # 下面的数据不更新计算梯度，也不会进行反向传播




