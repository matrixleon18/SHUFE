# encoding=GBK

"""
numpy�������
"""


import numpy as np

# �������������
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr * 2)
print(arr + 2)
print(arr/2)
arr *= 3            # ֱ���滻
print(arr)

# ���������������
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr + arr)
print(arr - arr)
print(arr * arr)
print(arr / arr)

arr2 = arr + 1
print(arr2 > arr)   # ������bool������

print(arr + arr2)
