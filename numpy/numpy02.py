# encoding=GBK

import numpy as np


nd_arr = np.arange(6)

print(nd_arr)

print(nd_arr.dtype)                 # ������������� int64

print(nd_arr.reshape(2, 3))         # ��һά����ת��2x3�ľ���

print(np.float16([0, 1]))           # ����һ��ָ���������͵�array

print(nd_arr.astype(float))         # ͨ��astypeת����������

print(np.float128.mro())            # ͨ��mro�鿴���͵ĸ���

print(np.issubdtype(nd_arr.dtype, np.floating))     # ͨ��issubdtype���ж��������� False

print(np.issubdtype(nd_arr.dtype, np.int64))        # True






