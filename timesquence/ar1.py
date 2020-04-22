# encoding=GBK


"""
ar�Իع�ģ�ͣ������ֵ�ͺ�ͼɢ��ͼ
"""
import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hs300_data = ts.get_hist_data('399300')

######################################################################
plt.plot(hs300_data['close'].to_list())                   # ԭʼ���ݵ�ͼ��
plt.show()

######################################################################
close_data = pd.Series(hs300_data['close'].to_list())       # ���̼�ת��list,��ת��series
pd.plotting.lag_plot(close_data)                            # ����ɢ��ͼ���ͺ�ͼ
plt.show()

######################################################################
values = pd.DataFrame(hs300_data['close'].to_list())    # �õ�һ��n��1�е�dataframe
print(values.shape)                                     # �����״ (608,1)
df = pd.concat([values.shift(1), values], axis=1)       # ����һ������һ�񣬰ѵڶ������ݰ����з�ʽ�ϲ�����
print(df.shape)                                         # �����״ (608,2)
df.columns = ['t-1', 't+1']                             # ������������

result = df.corr(method='pearson')                      # pearson:  ������������ݵ����ϵ������
                                                        # kendall:  ������������е����ϵ��������̫�ֲ�������
                                                        # spearman: �����Եģ�����̫���������ݵ����ϵ��
print(result)                                           # �����0.98���ӽ���1�߶������

######################################################################
pd.plotting.autocorrelation_plot(close_data)            # ���������ϵ��ͼ
plt.show()

plot_acf(close_data, lags=20)                           # ����ƫ�����ϵ��ͼ
plt.show()