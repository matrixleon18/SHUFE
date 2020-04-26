# encoding=GBK

"""
这是maptlotlib的基础用法
"""

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6), dpi=80)    # 创建一个 8*6 点（point）的图，并设置分辨率为 80

plt.subplot(1, 1, 1)                        # 创建一个新的 1*1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）

X = np.linspace(                            # 生成一个等差序列
    -np.pi,                                 # 序列的起点从 -3.14 开始
    np.pi,                                  # 序列的终点到 +3.14 结束
    256,                                    # 序列总数生成 256 个点
    endpoint=True                           # 如果是真，则一定包括stop，如果为False，一定不会有stop
)

C, S = np.cos(X), np.sin(X)                 # 分别把X的 cos,sin 的值赋给 C,S

plt.plot(                                   # 开始绘制cos线
    X,                                      # 点的x值
    C,                                      # 点的y值
    color='blue',                           # 用蓝色
    linewidth=1.0,                          # 线的宽度。float
    linestyle='-'                           # 实线/虚线/点 {’-’, ‘–-’, ‘:’, ‘-.’, ‘None’, ' ', '', 'solid', 'dotted' …}
)

plt.plot(                                   # 开始绘制sin线
    X,                                      # 点的x值
    S,                                      # 点的y值
    color='green',                          # 用绿色
    linewidth=1.0,                          # 线宽1.0
    linestyle='-'                           # 实线
)

plt.xlim(                                   # 设置横轴的上下限
    -4.0,                                   # 坐标从 -4 开始
    4.0                                     # 坐标到 4 结束
)

plt.xticks(                                 # 设置横轴的显示刻度
    np.linspace(-4, 4, 9, endpoint=True)    # 从-4到4显示9个刻度
)

plt.ylim(                                   # 设置纵轴的上下限
    -1.0,                                   # 坐标从 -1 开始
    1.0                                     # 坐标到  1 结束
)

plt.yticks(                                 # 设置纵轴的显示刻度
    np.linspace(-1, 1, 5, endpoint=True)    # 从-1到1显示5个刻度
)

plt.savefig("exercice.png", dpi=72)         # 把图存成文件

plt.show()                                  # 在屏幕上显示