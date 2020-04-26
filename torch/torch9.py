# encoding=GBK
"""
������Ĵ������洢����ȡ
"""


import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


torch.manual_seed(1)                                            # ΪCPU�������������������������ʹ�õõ��������ȷ����

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)          # ������һ��[100, 1]�����飻ֵ�Ǵӣ��������ģ������Ȳ�����
y = x.pow(2) + 0.2*torch.rand(x.size())                         # �ѣ�ƽ��һ�£��ټ���һ����С���������ΪNoise
x, y = Variable(x), Variable(y)                                 # ת���� Variable

def save_net():
    net1 = torch.nn.Sequential(                                 # ����һ������������
        torch.nn.Linear(1, 100),                                # ����ڵ���inputSize=1������ڵ���outputSize=100.init������
                                                                # self.weight=torch.Tensor(outputSize, inputSize)
                                                                # self.bias = torch.Tensor(outputSize)
                                                                # self.gradWeight = torch.Tensor(outputSize, inputSize)
                                                                # self.gradBias = torch.Tensor(outputSize)
                                                                # parent.__init() ���� gradInput, output, _type
                                                                # ����Linerһ���ǣ�������
        torch.nn.ReLU(inplace=True),                            # �����. inplace=True�Լ�����������Ӱ�졣
                                                                # ����in-place������Խ�ʡ�ڣ��ԣ��棬ͬʱ������ʡȥ����������ͷ��ڴ��ʱ�䡣���ǻ��ԭ�������ǣ�ֻҪ�������������
        torch.nn.Linear(100, 1)                                 # ���룱�������������
    )

    # print(list(net1.parameters()))

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.01)     # ��һ������Ҫ���Ż��Ĳ���������ʽ������Tensor����dict���ڶ������Ż�ѡ�����ѧϰ�ʡ�˥���ʵ�
                                                                # ����SGD�Ż���;optimizer����net1����parameters��ָ��;
                                                                # SGD�̳���Optimizer, ��Optimizer���__init__()��,
                                                                # ͨ��add_param_group()��ӽ���self.param_groups ����params���ֵ��
                                                                # ÿ��parameter�������ݶȣ�gradient��optimizer���԰��ݶ�Ӧ����ȥ����parameter

    loss_func = torch.nn.MSELoss()                              # ������ʧ����.
                                                                # ��prediction��y֮����бȶԣ��ػ�������loss function��������������ݶ�

    for epoch in range(1000):                                   # ��ʼ1000��epoch����ѵ��
        prediction = net1(x)                                    # �ѹ۲������ģ���еõ�Ԥ��ֵ
        loss = loss_func(prediction, y)                         # ͨ����ʧ����������۲�ֵ��Ԥ��ֵ�����
        optimizer.zero_grad()                                   # ����Ż����ϴ��������Ĳ����ݶ���Ϣ

        loss.backward()                                         # �����������練�򴫵ݻ�ȥ
                                                                # ���򴫲������������������·�ͽڵ㡣�ڵ���ڵ�֮������ϵ����˿��Է�����ʽ�����ݶȡ�

        optimizer.step()                                        # �Ż����Ϳ��Ը���loss����������������ݶȡ�����SGD����
                                                                # ����gradient�������������d.grad��loss.backward()�Զ��󵼻����Ѽ�����˲�����gradient
                                                                # ������Щparameter��gradient��parameter��ֵ���и��¡����ﲻ��Ҫ�����ݶȣ���Ϊ�ݶȵ�ָ���Ѿ����乹�캯���д����parameters�а���

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
