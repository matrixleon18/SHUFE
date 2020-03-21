# encoding=GBK

"""
如和使用梯度下降法来求解二元一次方程组的参数
"""
import numpy as np
import torch
from torch.autograd import Variable


np_data = np.arange(6).reshape(2,3)

torch_data = torch.from_numpy(np_data)

tensor_array = torch_data.numpy()

print(np_data)
print(torch_data)
print(tensor_array)

# abs
data = [-1,-2,3,-5,9,-9]
ten = torch.FloatTensor(data)

print(data)
print(ten)
print(torch.abs(ten))
print(np.abs(data))
print(torch.mean(ten))


data = [[1,2],[3,4]]
tens = torch.FloatTensor(data)

print(np.matmul(data, data))
print(torch.matmul(tens, tens))

print(np.array(tens))

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor)

print(tensor)
print(variable)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)

print(variable.__dir__())