# encoding=GBK

"""
ɢ��ͼ
"""

import numpy as np
import matplotlib.pyplot as plt

n = 1024

X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)          # ���ݸ����ĵ�Y��Xֵ�����Ӧ�㷴��ֵ

plt.scatter(
    X,                          # ���X����
    Y,                          # ���Y����
    s=75,                       # ��Ĵ�С
    c=T,                        # �����ɫ
    alpha=0.5                   # ���͸����
)

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()