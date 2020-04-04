# encoding=GBK


"""
ar�Իع�ģ�ͣ������ֵ�ͺ�ͼɢ��ͼ
"""
import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hs300_data = ts.get_hist_data('399300')
close_data = pd.Series(hs300_data['close'].to_list())

pd.plotting.lag_plot(close_data)
plt.show()

# pd.plotting.autocorrelation_plot(close_data)
# plt.show()