# encoding=GBK

"""
��������� CNN
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
DOWNLOAD_MNIST = False                                  # ��һ�����а��������True���������ݼ����Ժ����False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get train data
train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,                                         # ����load���� MNIST �����train���ݡ���60K
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)


# Get test data
test_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=False
)


# show the test data
# print(train_data.data.size())                           # ���ǵ���� MNIST ѵ�����ݴ�С [60000, 28, 28]��ʽ
# print(train_data.targets.size())                        # ���ǵ���� MNIST ���ݶ�Ӧ����ʵ���� [60000]��ʽ
# print(train_data.data[0])                               # Ŀǰ��ͼƬ���ݻ�������(0,255)������
# plt.imshow(train_data.data[0].numpy())                  # �� MNIST ��һ��ͼ����������
# print(train_data.targets[0])                            # �� MNITST ��һ��ͼ���ע���������֡�Ҳ��������Ԥ���Ŀ��
# plt.show()

train_loader = loader.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

with torch.no_grad():                                     # ��������ݲ����¼����ݶȣ���Ϊ�����Լ������������㣬���ü�¼����ͼͼ�����򴫲�������Ԥ��
    test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:2000]/255    # ����/255Ϊ��ѹ����(0,1)������
    test_y = test_data.targets[:2000]                                                               # ������2000������


# print(test_x.size())                                        # �������ݱ����[2000, 1, 28, 28]��ʽ
# print(test_x[0][0])                                         # ����ͼƬ���ݱ�ѹ������(0,1)������
# plt.imshow(test_x[0][0].numpy())                            # ����ͼƬ������ʾû�б仯
# plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(                                      # 2D�����. ���������size = channel*W*H = 1*28*28
                in_channels=1,                              # �����ά��
                out_channels=16,                            # �����filter������Ҳ����16���ͼ
                kernel_size=5,                              # �������ڵĴ�С��5x5����
                stride=1,                                   # ÿ���ƶ��Ĳ���
                padding=2,                                  # ͼ��߽粹��������������Ϊ(kernel_size-1)/2=2
            ),                                              # ������� size = 16*28*28
            nn.ReLU(),                                      # �����. ���ͬ��
            nn.MaxPool2d(
                kernel_size=2,                              # ��2x2������ѡȡ����ֵ
            )                                               # �������size=16*14*14 ����Ϊ��ȡ����
        )

        self.conv2 = nn.Sequential(                         # ��������size=16*14*14
            nn.Conv2d(16, 32, 5, 1, 2),                     # �ο������ Conv2d ����; �������size=32*14*14
            nn.ReLU(),                                      # �������size=32*14*14
            nn.MaxPool2d(2)                                 # �������size=32*7*7
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # x.to(device)
        x = self.conv1(x)
        x = self.conv2(x)                           # (batch, 32, 7, 7)
        output = x.view(x.size(0), -1)                   # (batch, 32*7*7)
        output = self.out(output)
        return output


cnn = CNN()                                                 # ����һ���������ʵ��
# print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)       # �Ż������������������еĲ���
loss_func = nn.CrossEntropyLoss()                           # ��ʧ����

for epoch in range(EPOCH):                                  # ��ʼѵ����
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