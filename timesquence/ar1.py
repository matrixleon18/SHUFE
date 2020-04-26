# encoding=GBK


"""
ar自回归模型－－相关值滞后图散点图
"""
import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hs300_data = ts.get_hist_data('399300')

######################################################################
plt.plot(hs300_data['close'].to_list())                   # 原始数据的图像
plt.show()

######################################################################
close_data = pd.Series(hs300_data['close'].to_list())       # 收盘价转成list,再转成series
pd.plotting.lag_plot(close_data)                            # 绘制散点图的滞后图
plt.show()

######################################################################
values = pd.DataFrame(hs300_data['close'].to_list())    # 得到一个n行1列的dataframe
print(values.shape)                                     # 打出形状 (608,1)
df = pd.concat([values.shift(1), values], axis=1)       # 将第一列下移一格，把第二组数据按照列方式合并起来
print(df.shape)                                         # 打出形状 (608,2)
df.columns = ['t-1', 't+1']                             # 给这两列命名

result = df.corr(method='pearson')                      # pearson:  即针对线性数据的相关系数计算
                                                        # kendall:  即针对无序序列的相关系数，非正太分布的数据
                                                        # spearman: 非线性的，非正太分析的数据的相关系数
print(result)                                           # 相关性0.98　接近于1高度正相关

######################################################################
pd.plotting.autocorrelation_plot(close_data)            # 绘制自相关系数图
plt.show()

plot_acf(close_data, lags=20)                           # 绘制偏自相关系数图
plt.show()