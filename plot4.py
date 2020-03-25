# encoding=GBK

"""
散点图
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 50, endpoint=True) + np.random.rand(50)
Y = X ** 2 + 3

plt.scatter(X, Y)

plt.show()
