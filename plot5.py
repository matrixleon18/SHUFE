# encoding=GBK

"""
���Ƶȸ���ͼ
"""
import numpy as np
import matplotlib.pyplot as plt

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X, Y = np.meshgrid(x, y)    # meshgrid���������������������


def f(x, y):
    """
    ������������߶�ֵ ����contour��������ɫ�ӽ�ȥ λ�ò�������Ϊx,y,f(x,y)��͸����Ϊ0.75������f(x,y)��ֵ��Ӧ��camp֮��
    :param x:
    :param y:
    :return:
    """
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.cm.hot)   # ��ʾ�ȸ��߷ֳɶ��ٷ� alpha��ʾ͸���� cmap��ʾcolor map

# ʹ��plt.contour�������еȸ��߻��� ��������Ϊx,y,f(x,y)����ɫѡ���ɫ���������Ϊ0.5
C=plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=0.5)

# ʹ��plt.clabel��Ӹ߶���ֵ inline�����Ƿ�label���������棬�����СΪ10
plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())                      # ����������
plt.yticks(())
plt.show()