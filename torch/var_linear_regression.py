# encoding=GBK


import numpy as np
import torch as t
from torch.autograd import Variable as V
import matplotlib.pyplot as plt


# np.random.seed(1)
# x = np.random.rand(10)
# y = np.random.rand(10)
# c = np.random.rand(10)
# area = (30 * np.random.rand(10))**2
#
# plt.scatter(x, y, s=area, c=c, alpha=0.5)
# plt.show()


t.manual_seed(1000)


def get_fake_data(batch_size=8):
    """产生随机数据： y = x*2 + 3  再加上一些噪声"""
    x = t.rand(batch_size, 1) * 20
    y = x * 2 + (1 + t.randn(batch_size, 1)) * 3
    return x, y


x, y = get_fake_data()
print(x.squeeze().numpy())
print(y.squeeze().numpy())
plt.plot(x.squeeze().numpy(), y.squeeze().numpy())
# plt.show()