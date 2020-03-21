# encoding=GBK

"""
循环神经网络分类 MNIST
"""


import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch.utils.data.dataloader

# 超参数
EPOCH = 1                       # 训练全数据集的次数
BATCH_SIZE = 64                 # 每次取的batch的数据大小
TIME_STEP = 28                  # RNN考虑的时间节点。因为每次抽取图像一行数据处理。这里就是图像的高度。
INPUT_SIZE = 28                 # RNN每次处理的数据量。因为是一次一行数据，那就是图像的宽度
LR = 0.001
DOWLOAD_MNIST = False
HIDDEN_LAYER_FEATURES = 128

# 准备训练和测试数据
# with torch.no_grad():
train_data = datasets.MNIST("./mnist", train=True, transform=transforms.ToTensor(), download=DOWLOAD_MNIST)
test_data = datasets.MNIST("./mnist", train=False, transform=transforms.ToTensor(), download=DOWLOAD_MNIST)
test_x = Variable(test_data.data).type(torch.FloatTensor)[:1000]/255
test_y = test_data.targets.numpy().squeeze()[:1000]

train_loader = torch.utils.data.dataloader.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# 定义RNN网络结构的类
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # LSTM接受的输入数据格式必须是3维的Tensor
        # 第一维度是batch_size，就是一次性输入给了RNN多少条句子，或者是一次多少个时间单位的股票数据。
        # 第二维度是序列sequence结构，也就是序列的个数，如果是文章就是每个句子长度。如果是股票数据，那就是一次来了多少条数据。明确这个层中有多少个确定的单元来处理输入数据
        # 第三维度是input_size,也就是输入的元素个数。每个具体的单词用多少维向量来表示，或者股票一个具体时刻采集多少feature,比如最高价，最低价，5日均线等
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,                      # x的特征维度
            hidden_size=HIDDEN_LAYER_FEATURES,          # 隐藏层的特征维度
            num_layers=1,                               # RNN 具有的细胞的数量，也就是lstm的隐藏层的数量
            batch_first=True,                           # LSTM接受的输入数据格式必须是3维的。如果是True就意味着输入输出数据格式为 (batch, seq, feature)
            # dropout=0,                                  # 除了最后一层，每一层的输出都要dropout,默认为0
            # bidirectional=False,                        # True则为双向LSTM，默认为False
        )

        self.out = nn.Linear(HIDDEN_LAYER_FEATURES, 10) # 输入的是64特征维度，输出是10特征维度.因为是0~9共计10个数字的分类

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)           # 现在的ｘ是(batch, time_step, input_size)；没有第一个隐藏状态就是None.
                                                        # H0~Hn就是每个神经元根据输入值和上一时刻的状态值合起来产生的本时刻的状态值. h_n shape (n_layers, batch, hidden_size)
                                                        # C0~Cn就是开关。决定每个神经元的当前隐藏状态值是否会影响下一时刻的处理.     h_c shape (n_layers, batch, hidden_size)
                                                        # r_out存贮了之前所有的输出                                             r_out shape (batch, time_step, output_size)
        out = self.out(r_out[:, -1, :])                 # 选取最后一个时刻的输出。也就是整个RNN的输出[batch, sequence[-1], input_size]
        return out


# 生成一个RNN的实例。
rnn = RNN()
# print(rnn)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
# optimizer = torch.optim.RMSprop(rnn.parameters(), alpha=0.9)
loss_func = torch.nn.CrossEntropyLoss()

# 开始训练
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28, 28)          # 把输入的数据再转成(batch, sequence, input)的格式
        # b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 用训练过的模型跑一下测试数据
        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / test_y.size
            print("Epoch: {}, loss: {}, accuracy: {}".format(epoch, round(float(loss.data), 3), accuracy))

test_output = rnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print("pred: ", pred_y)
print("real: ", test_y[:10])