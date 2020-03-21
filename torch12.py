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
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False                                  # ��һ�����а��������True���������ݼ����Ժ����False

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

test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:2000]/255    # ����/255Ϊ��ѹ����(0,1)������
test_y = test_data.targets[:2000]

print(test_x.size())                                        # �������ݱ����[2000, 1, 28, 28]��ʽ
print(test_x[0][0])                                         # ����ͼƬ���ݱ�ѹ������(0,1)������
plt.imshow(test_x[0][0].numpy())                            # ����ͼƬ������ʾû�б仯
plt.show()
# with torch.no_grad:                                     # ��������ݲ����¼����ݶȣ�Ҳ������з��򴫲�




