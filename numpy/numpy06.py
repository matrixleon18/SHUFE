# encoding='GBK'

"""
统计方法
"""


import numpy as np

arr = np.array([[np.NaN, 0, 1, 2, 3],
                [4, 5, np.NaN, 6, 7],
                [8, 9, 10, np.NaN, 11]])

print(np.amax(arr))         # 返回数组的最大值或沿轴的最大值
# nan
print(np.nanmax(arr))       # 返回数组中除NaN以外的最大值
# 11.0
print(np.nanmin(arr))       # 返回数组中除NaN以外的最小值
# 0.0
print(np.nanmean(arr))      # 返回数组中除NaN以外的均值
# 5.5
print(np.nanmedian(arr))    # 返回数组中除NaN以外的中位数
# 5.5
print(np.nanstd(arr))       # 返回数组中除NaN以外的标准差 方差开根号
# 3.452052529534663
print(np.nanvar(arr))       # 返回数组中除NaN以外的方差 用来度量随机变量和其数学期望（即均值）之间的偏离程度
# 11.916666666666666
# True/False会被处理成1/0
arr = np.array([0.1, 0.2, 0.3, True, 0.4, False])
print(np.nanmax(arr))
# 1.0
print(np.nanmin(arr))
# 0.0