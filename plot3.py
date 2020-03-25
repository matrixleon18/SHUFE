# encoding=GBK

"""
双轴标轴图
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(111)
ax1.plot(X, C, 'b-', label='TPS', lw=2)
ax2 = ax1.twinx()
ax2.plot(X, S*100, 'r-', label='OK', lw=2)

plt.show()