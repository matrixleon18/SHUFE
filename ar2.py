# encoding=GBK

"""
�Իع�ģ�ͣ�������Ե�ֵ
"""
import tushare as ts
import pandas as pd

hs300_data = ts.get_hist_data('399300')                 # �õ�����300��������
values = pd.DataFrame(hs300_data['close'].to_list())    # �õ�һ��n��1�е�dataframe
print(values.shape)                                     # �����״ (608,1)
df = pd.concat([values.shift(1), values], axis=1)       # ����һ������һ�񣬰ѵڶ������ݰ����з�ʽ�ϲ�����
print(df.shape)                                         # �����״ (608,2)
df.columns = ['t-1', 't+1']                             # ������������

result = df.corr(method='pearson')                      # pearson:  ������������ݵ����ϵ������
                                                        # kendall:  ������������е����ϵ��������̫�ֲ�������
                                                        # spearman: �����Եģ�����̫���������ݵ����ϵ��
print(result)                                           # �����0.98���ӽ���1�߶������
