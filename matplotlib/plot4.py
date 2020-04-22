# encoding=GBK

"""
散点图
"""

import numpy as np
import matplotlib.pyplot as plt

n = 1024

X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)          # 根据给出的点Y，X值计算对应点反切值

plt.scatter(
    X,                          # 点的X坐标
    Y,                          # 点点Y坐标
    s=75,                       # 点的大小
    c=T,                        # 点的颜色
    alpha=0.5                   # 点的透明度
)

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()