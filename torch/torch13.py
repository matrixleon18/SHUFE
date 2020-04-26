# encoding=GBK

"""
ѭ����������� MNIST
"""


import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch.utils.data.dataloader

# ������
EPOCH = 1                       # ѵ��ȫ���ݼ��Ĵ���
BATCH_SIZE = 64                 # ÿ��ȡ��batch�����ݴ�С
TIME_STEP = 28                  # RNN���ǵ�ʱ��ڵ㡣��Ϊÿ�γ�ȡͼ��һ�����ݴ����������ͼ��ĸ߶ȡ�
INPUT_SIZE = 28                 # RNNÿ�δ��������������Ϊ��һ��һ�����ݣ��Ǿ���ͼ��Ŀ��
LR = 0.001
DOWLOAD_MNIST = False
HIDDEN_LAYER_FEATURES = 128

# ׼��ѵ���Ͳ�������
# with torch.no_grad():
train_data = datasets.MNIST("./mnist", train=True, transform=transforms.ToTensor(), download=DOWLOAD_MNIST)
test_data = datasets.MNIST("./mnist", train=False, transform=transforms.ToTensor(), download=DOWLOAD_MNIST)
test_x = Variable(test_data.data).type(torch.FloatTensor)[:1000]/255
test_y = test_data.targets.numpy().squeeze()[:1000]

train_loader = torch.utils.data.dataloader.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# ����RNN����ṹ����
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # LSTM���ܵ��������ݸ�ʽ������3ά��Tensor
        # ��һά����batch_size������һ�����������RNN���������ӣ�������һ�ζ��ٸ�ʱ�䵥λ�Ĺ�Ʊ���ݡ�
        # �ڶ�ά��������sequence�ṹ��Ҳ�������еĸ�������������¾���ÿ�����ӳ��ȡ�����ǹ�Ʊ���ݣ��Ǿ���һ�����˶��������ݡ���ȷ��������ж��ٸ�ȷ���ĵ�Ԫ��������������
        # ����ά����input_size,Ҳ���������Ԫ�ظ�����ÿ������ĵ����ö���ά��������ʾ�����߹�Ʊһ������ʱ�̲ɼ�����feature,������߼ۣ���ͼۣ�5�վ��ߵ�
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,                      # x������ά��
            hidden_size=HIDDEN_LAYER_FEATURES,          # ���ز������ά��
            num_layers=1,                               # RNN ���е�ϸ����������Ҳ����lstm�����ز������
            batch_first=True,                           # LSTM���ܵ��������ݸ�ʽ������3ά�ġ������True����ζ������������ݸ�ʽΪ (batch, seq, feature)
            # dropout=0,                                  # �������һ�㣬ÿһ��������Ҫdropout,Ĭ��Ϊ0
            # bidirectional=False,                        # True��Ϊ˫��LSTM��Ĭ��ΪFalse
        )

        self.out = nn.Linear(HIDDEN_LAYER_FEATURES, 10) # �������64����ά�ȣ������10����ά��.��Ϊ��0~9����10�����ֵķ���

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)           # ���ڵģ���(batch, time_step, input_size)��û�е�һ������״̬����None.
                                                        # H0~Hn����ÿ����Ԫ��������ֵ����һʱ�̵�״ֵ̬�����������ı�ʱ�̵�״ֵ̬. h_n shape (n_layers, batch, hidden_size)
                                                        # C0~Cn���ǿ��ء�����ÿ����Ԫ�ĵ�ǰ����״ֵ̬�Ƿ��Ӱ����һʱ�̵Ĵ���.     h_c shape (n_layers, batch, hidden_size)
                                                        # r_out������֮ǰ���е����                                             r_out shape (batch, time_step, output_size)
        out = self.out(r_out[:, -1, :])                 # ѡȡ���һ��ʱ�̵������Ҳ��������RNN�����[batch, sequence[-1], input_size]
        return out


# ����һ��RNN��ʵ����
rnn = RNN()
# print(rnn)

# �����Ż�������ʧ����
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
# optimizer = torch.optim.RMSprop(rnn.parameters(), alpha=0.9)
loss_func = torch.nn.CrossEntropyLoss()

# ��ʼѵ��
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28, 28)          # �������������ת��(batch, sequence, input)�ĸ�ʽ
        # b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ��ѵ������ģ����һ�²�������
        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / test_y.size
            print("Epoch: {}, loss: {}, accuracy: {}".format(epoch, round(float(loss.data), 3), accuracy))

test_output = rnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print("pred: ", pred_y)
print("real: ", test_y[:10])