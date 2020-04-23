# encoding=GBK

"""
numpy数组计算
"""


import numpy as np

# 数组与标量运算
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr * 2)
print(arr + 2)
print(arr/2)
arr *= 3            # 直接替换
print(arr)

# 数组与数组的运算
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr + arr)
print(arr - arr)
print(arr * arr)
print(arr / arr)

arr2 = arr + 1
print(arr2 > arr)   # 生成了bool的数组

print(arr + arr2)
