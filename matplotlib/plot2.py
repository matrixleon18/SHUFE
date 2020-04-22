# encoding=GBK

"""
���������Ƶ��м�
"""

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 6), dpi=80)   # ͼ��ˮƽ��������һ��

plt.subplot(1, 1, 1)
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C, color='blue',linewidth=10.0, linestyle='-', label='cos')                 # ���ϱ�ǩ
plt.plot(X, S, color='green', linewidth=10.0, linestyle='-', label='sin', alpha=0.7)    # ����͸����
plt.xlim(X.min()*1.1, X.max()*1.1)
plt.xticks(np.linspace(-4, 4, 13, endpoint=True))
plt.xlabel('x-label')
plt.ylim(C.min()*1.1, C.max()*1.1)
plt.yticks(np.linspace(-1, 1, 5, endpoint=True))


ax = plt.gca()                                          # get current axis
ax.spines['right'].set_color('none')                    # �ұ��߲���ʾ
ax.spines['top'].set_color('none')                      # �ϱ��߲���ʾ
ax.xaxis.set_ticks_position('bottom')                   # ��x��
ax.spines['bottom'].set_position(('data', 0))           # ��x���Ƶ��м�
ax.yaxis.set_ticks_position('left')                     # ��y��
ax.spines['left'].set_position(('data', 0))             # ��y���Ƶ��м�


plt.legend(loc='upper left')                            # �����Ͻ���ʾ��ǩ

plt.show()
