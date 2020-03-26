# encoding=GBK

"""
把坐标轴移到中间
"""

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 6), dpi=80)   # 图在水平方向拉升一点

plt.subplot(1, 1, 1)
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C, color='blue',linewidth=10.0, linestyle='-', label='cos')                 # 加上标签
plt.plot(X, S, color='green', linewidth=10.0, linestyle='-', label='sin', alpha=0.7)    # 设置透明度
plt.xlim(X.min()*1.1, X.max()*1.1)
plt.xticks(np.linspace(-4, 4, 13, endpoint=True))
plt.xlabel('x-label')
plt.ylim(C.min()*1.1, C.max()*1.1)
plt.yticks(np.linspace(-1, 1, 5, endpoint=True))


ax = plt.gca()                                          # get current axis
ax.spines['right'].set_color('none')                    # 右边线不显示
ax.spines['top'].set_color('none')                      # 上边线不显示
ax.xaxis.set_ticks_position('bottom')                   # 绑定x轴
ax.spines['bottom'].set_position(('data', 0))           # 把x轴移到中间
ax.yaxis.set_ticks_position('left')                     # 绑定y轴
ax.spines['left'].set_position(('data', 0))             # 把y轴移到中间


plt.legend(loc='upper left')                            # 在左上角显示标签

plt.show()
