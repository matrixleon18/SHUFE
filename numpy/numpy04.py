# encoding="GBK"

"""
numpy 的深拷贝，浅拷贝，不拷贝
"""

import numpy as np

# 完全不拷贝，就是引用
a = np.arange(5)
print(a)
# [0 1 2 3 4]
b = a
print(b)
# [0 1 2 3 4]
b[0] = 1
print(b)
# [1 1 2 3 4]
print(a)
# [1 1 2 3 4]
print(b is a)
# True

# 浅拷贝
a = np.arange(5)
print(a)
# [0 1 2 3 4]
c = a.view()
print(c)
# [0 1 2 3 4]
c[0] = 1
print(c)
# [1 1 2 3 4]
print(a)
# [1 1 2 3 4]
print(c is a)
# False
print(c.base is a)
# True
print(a.shape)
# (5,)
print(c.shape)
# (5,)
c.shape = (1, 5)
print(c.shape)
# (1, 5)
print(a.shape)
# (5,)
print(c)
# [[1 1 2 3 4]]
print(a)
# [1 1 2 3 4]

# 深拷贝。这就是另外一个实例
a = np.arange(5)
d = a.copy()
print(a)
# [0 1 2 3 4]
print(d)
# [0 1 2 3 4]

print(d is a)
# False

d[0] = 1
print(d)
# [1 1 2 3 4]
print(a)
# [0 1 2 3 4]

d.shape = (1, 5)
print(d)
# [[1 1 2 3 4]]
print(a)
# [0 1 2 3 4]
