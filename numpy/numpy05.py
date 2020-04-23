# encoding='BGK'

"""
切片和索引
"""

import numpy as np

# 一维切片
a = np.arange(10)

s = slice(2, 7, 2)          # 从2到7，每隔一个数字取一个
print(a[s])
# [2 4 6]
print(a[4: -3])             # 从4开始到倒数第三个为止，不包含倒数第三个(7)
# [4 5 6]
print(a[:7])                # 从开始到第八个。不包含第八个。
# [0 1 2 3 4 5 6]
print(a[9: 4: -2])          # 从第9个开始倒着隔一个取一个直到第4个
#  [9 7 5]
print(a[:: 2])              # 取偶数位的值

# 多维切片
y = np.arange(16).reshape(4, 4)     # 4x4的数组
print(y[..., 2])            # 取第二列
# [ 2  6 10 14]
print(y[[1]])               # 取第二行
# [[4 5 6 7]]
print(y[1:2])               # 取第二行
# [[4 5 6 7]]
print(y[:2, :2])            # 取前两行前两列
# [[0 1]
#  [4 5]]
print(y[::2, -2])           # 步长为2的行，倒数第二列
# [ 2 10]

# 索引
arr = np.arange(16).reshape(4, 4)
print(arr[1])               # 第二行的索引
# [4 5 6 7]
print(arr[1, 1])            # 第二行第二列数字的索引。等效于arr[1][1],但后者更慢
# 5
print(arr[:3, [0, 1]])      # 前三行的第一第二列值
# [[0 1]
#  [4 5]
#  [8 9]]

# 布尔索引
cities = np.array(['bj', 'dl', 'sh', 'gz', 'cd'])
data = np.arange(20).reshape(5, 4)
print(data)
print(cities == 'cd')
# [False False False False  True]
print(data[cities == 'cd'])
# [[16 17 18 19]]               # 找到了对应为true的那一行
print(data[cities == 'cd', 1])  # 找到了对应的行的第二个数字
# [17]
