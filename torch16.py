# encoding=GBK

"""
AutoEncoder�Ա���
"""

import torch
import torchvision
from torch.autograd.variable import Variable
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

train_data = datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = torch.utils.data.dataloader.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# print(train_data.data.size())               # [60000, 28, 28] �����60K��28X28ͼ��ѵ����
# print(train_data.targets.size())            # [60000] ����ÿ��ͼ���Ӧ�ķ���[0 ~ 9]����
# plt.imshow(train_data.data[2], cmap='gray') # colormap: gray, 0~255���Ҷ�,0��ɫ1��ɫ; gray_r,��ת; seismic,���˷�ɢ��ɫͼ; rainbow,�ʺ�ͼ;
# plt.title(train_data.targets[2].numpy())
# plt.show()

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(                         # ����ѹ����
            torch.nn.Linear(in_features=28*28, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=32, out_features=3)         # ���3��5��ά�ȶ�loss��𲻴�
        )

        self.decoder = torch.nn.Sequential(                         # ���ǽ�ѹ��
            torch.nn.Linear(in_features=3, out_features=32),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=64, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=28*28),
            # torch.nn.Sigmoid()                                   # �������(0,1)�ĸ���
            # torch.nn.Tanh()                                      # �������(0,1)�ĸ��ʡ���Sigmod���
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


auto_encoder = AutoEncoder()
optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

view_data = Variable(train_data.data[0:9].view(-1, 28 * 28).type(torch.FloatTensor) / 255.)
view_target = Variable(train_data.targets[0:9])

f, a = plt.subplots(1, 9, figsize=(9, 2))

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28*28))       # batch x, re-shape to (batch, 28*28)
        b_y = Variable(x.view(-1, 28*28))       # batch y, re-shape to (batch, 28*28)
        b_label = Variable(y)

        encoded, decoded = auto_encoder(b_x)

        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch: {} step: {} loss: {}".format(epoch, step, loss.data.numpy()))

        # if step % 5 == 0:
        #     plt.ion()
        #     plt.imshow(decoded.view(-1, 28, 28).data[0], cmap='gray')
        #     plt.title(b_label[0].numpy())
        #     plt.show()
        #     plt.pause(0.01)

        if step % 500 == 0:
            plt.ion()
            _, decoded_data = auto_encoder(view_data)
            # f, a = plt.subplots(1, 9, figsize=(9, 2))

            for i in range(9):
                # a[1][i].clear()
                decoded_data = decoded_data.view(-1, 28, 28)
                a[i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[i].set_title(view_target.data.numpy()[i])
                a[i].set_xticks(())
                a[i].set_yticks(())

            plt.show()
            plt.pause(0.001)