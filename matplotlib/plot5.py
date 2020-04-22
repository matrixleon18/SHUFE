# encoding=GBK

"""
绘制等高线图
"""
import numpy as np
import matplotlib.pyplot as plt

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X, Y = np.meshgrid(x, y)    # meshgrid从坐标向量返回坐标矩阵


def f(x, y):
    """
    函数用来计算高度值 利用contour函数把颜色加进去 位置参数依次为x,y,f(x,y)，透明度为0.75，并将f(x,y)的值对应到camp之中
    :param x:
    :param y:
    :return:
    """
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.cm.hot)   # 表示等高线分成多少份 alpha表示透明度 cmap表示color map

# 使用plt.contour函数进行等高线绘制 参数依次为x,y,f(x,y)，颜色选择黑色，线条宽度为0.5
C=plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=0.5)

# 使用plt.clabel添加高度数值 inline控制是否将label画在线里面，字体大小为10
plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())                      # 隐藏坐标轴
plt.yticks(())
plt.show()