# encoding=GBK


"""
利用torch.autograd.grad求二阶导
"""


import torch
from torch import autograd

###########################################
# 自动求导
# a,b,c的值分别是１，２，３
# x 的值是　１
###########################################
x = torch.tensor(1)
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = torch.tensor(3.0, requires_grad=True)

y = a**2 * x + b * x + c

print('before: ', a.grad, b.grad, c.grad)
grades = autograd.grad(y, [a, b, c])
print('after: ', grades[0], grades[1], grades[2])