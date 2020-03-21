# encoding=GBK

"""
神经网络优化器
"""

import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

# hyper-parameter
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x

# Generate the data
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(x.size()))

# show data
# plt.scatter(x.numpy(), y.numpy())
# plt.show()

# Define the torch data set
torch_data_set = torch.utils.data.TensorDataset(x, y)

# load the torch data set
data_loader = torch.utils.data.DataLoader(dataset=torch_data_set, batch_size=BATCH_SIZE, shuffle=True)

# Create 4 nn
net_SGD      = Net()
net_Momentum = Net()
net_RMSprop  = Net()
net_Adam     = Net()

nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

# create 4 optimizers
opt_SGD      = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop  = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam     = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

opts = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()

loss_his = [[], [], [], []]

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(data_loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net, opt, l_his in zip(nets, opts, loss_his):
            output = net(b_x)
            loss = loss_func(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.item())

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']

for index, l_his in enumerate(loss_his):
    plt.plot(l_his, label=labels[index])

plt.legend(loc='best')
plt.xlabel("steps")
plt.ylabel("loss")
plt.ylim(0, 0.2)
plt.show()

