# encoding=GBK


"""
ar�Իع�ģ�ͣ������ֵ�ͺ�ͼɢ��ͼ
"""
import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hs300_data = ts.get_hist_data('399300')

# plt.plot(hs300_data['close'].to_list())                   # ԭʼ���ݵ�ͼ��
# plt.show()

close_data = pd.Series(hs300_data['close'].to_list())       # ���̼�ת��list,��ת��series
pd.plotting.lag_plot(close_data)                            # ����ɢ��ͼ���ͺ�ͼ
plt.show()