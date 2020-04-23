# encoding=GBK

import numpy as np


nd_arr = np.arange(6)

print(nd_arr)

print(nd_arr.dtype)                 # 输出数据类型是 int64

print(nd_arr.reshape(2, 3))         # 将一维数组转成2x3的矩阵

print(np.float16([0, 1]))           # 创建一个指定数据类型的array

print(nd_arr.astype(float))         # 通过astype转换数据类型

print(np.float128.mro())            # 通过mro查看类型的父类

print(np.issubdtype(nd_arr.dtype, np.floating))     # 通过issubdtype来判断数据类型 False

print(np.issubdtype(nd_arr.dtype, np.int64))        # True






