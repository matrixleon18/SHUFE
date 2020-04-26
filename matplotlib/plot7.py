# encoding=GBK

"""
绘制3D图形
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()                      # 定义一个画布
ax = Axes3D(fig)                        # 添加3D坐标系

X = np.arange(-4, 4, 0.2)
Y = np.arange(-4, 4, 0.2)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)                           # 这是高度值

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

ax.contour(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))  # 图形投影一下

ax.set_zlim(-2, 2)

plt.show()
