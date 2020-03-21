# encoding=GBK
"""
���������粢��ͼ
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

NUM_TOTAL_POINTS = 40

n_data = torch.ones(int(NUM_TOTAL_POINTS/2), 2)                                     # ����һȺ100���㣬���ݻ�׼����(1,1)

x0 = torch.normal(2*n_data, 1)                                                      # �Ե����������Ծ�ֵΪ������Ϊ���ķֲ�
y0 = torch.zeros(int(NUM_TOTAL_POINTS/2))                                           # ��һ��ĵ������Ϊ��

x1 = torch.normal(-2*n_data, 1)                                                     # �Ե���������Ծ�ֵΪ��������Ϊ���ķֲ�
y1 = torch.ones(int(NUM_TOTAL_POINTS/2))                                             # ��һ��ĵ���Ϊ��

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)                              # ��x0��x1����ά�ȣ�����ƴ�ӣ����ţ�
y = torch.cat((y0, y1), ).type(torch.LongTensor)                                # ��y0��y1����ƴ��

x, y = Variable(x), Variable(y)                                                 # ��x,y��ת����Variable


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


net = Net(2, 10, 2)                                         # ��������ֵ[������,������]�����onehot���������ֵ[0,1]��[1,0]��ʾ����
# print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()                     # CrossEntropyLoss/��������ʧ:������Է����������ʧ�������ر��Ƕ�������⡣������Ǹ���

for t in range(100):
        out = net(x)                                        # �������[200,2]��tensor,��Ϊ��200����,ÿ������������ֵ����[-3,4],
        loss = loss_func(out, y)                            # ���㽻����,����tensor(0.3932, grad_fn=<NllLossBackward>)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 2 == 0:
            plt.cla()
            out_softmax = torch.softmax(out, -2)            # �������һ����(0,1)�У�����[20,2]��tensor���Ǹ���,�к�Ϊ��
            ret = torch.max(out_softmax, 1)                 # ȡ��ÿһ�е����ֵ��������,���tensor(valuns=[20], indics=[20])
            prediction = ret[1]                             # �õ�ÿһ�����ֵ������ [20]
            pred_y = prediction.data.numpy().squeeze()      # ת����numpy���ݣ���ת��һάarray[20]�������Ԥ��ģ�����[1,0,0,...]
            target_y = y.data.numpy()                       # ����ʵ�ʵģ�����
            plt.scatter(x.data.numpy()[:, 0],               # ������X����
                        x.data.numpy()[:, 1],               # ������y����
                        c=pred_y,                           # ���ǵ����ɫ����ͬ�����ִ����˲�ͬ����ɫ��
                        s=100,                              # ��Ĵ�С
                        lw=0)                               # �߿�
            accuracy = sum(pred_y == target_y) / NUM_TOTAL_POINTS   # ����Ԥ����ȷ�ĸ���
            plt.text(0, -3, "Round: {} Accuracy: {}".format(t, accuracy))
            plt.pause(0.1)


plt.ioff()
plt.show()