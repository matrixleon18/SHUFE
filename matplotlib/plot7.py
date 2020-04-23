# encoding=GBK

"""
����3Dͼ��
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()                      # ����һ������
ax = Axes3D(fig)                        # ���3D����ϵ

X = np.arange(-4, 4, 0.2)
Y = np.arange(-4, 4, 0.2)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)                           # ���Ǹ߶�ֵ

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

ax.contour(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))  # ͼ��ͶӰһ��

ax.set_zlim(-2, 2)

plt.show()
