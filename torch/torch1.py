# encoding=GBK


"""
比较CPU和GPU模式下Torch速度
"""

import torch
import time

############################################
# this is torch CPU mode for matrix operation
############################################
a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print("CPU mode time cost : {}".format(t1-t0))
print(a.device, t1-t0, c.norm(2))

############################################
# this is GPU mode for matrix operation
# This include some torch initiation time
############################################
device = torch.device('cuda')
a = a.to(device)
b = b.to(device)
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print("GPU mode time cost : {}".format(t1-t0))
print(a.device, t1-t0, c.norm(2))

############################################
# this is real torch GPU mode for matrix operation
############################################
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print("GPU mode time cost : {}".format(t1-t0))
print(a.device, t1-t0, c.norm(2))

