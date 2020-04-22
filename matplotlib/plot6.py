# encoding=GBK

"""
绘制条形图
"""

import numpy as np
import matplotlib.pyplot as plt

n = 12

# 基本图形
X = np.arange(n)
Y1 = (1-X/float(n))*np.random.uniform(0.5, 1, n)
Y1 = (1-X/float(n))*np.random.uniform(0.5, 1, n)
plt.bar(X, +Y1, facecolor='red', edgecolor='white')
plt.bar(X, -Y1, facecolor='green', edgecolor='black')


# 标记值
for x, y in zip(X, Y1):             # zip 表示可以传递两个值
    plt.text(x+0.2, y+0.05, '%.2f'%y,ha='center', va='bottom')

plt.show()