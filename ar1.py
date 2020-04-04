# encoding=GBK


"""
ar自回归模型－－相关值滞后图散点图
"""
import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hs300_data = ts.get_hist_data('399300')

# plt.plot(hs300_data['close'].to_list())                   # 原始数据的图像
# plt.show()

close_data = pd.Series(hs300_data['close'].to_list())       # 收盘价转成list,再转成series
pd.plotting.lag_plot(close_data)                            # 绘制散点图的滞后图
plt.show()