# encoding=GBK

"""
自回归模型－－相关性的值
"""
import tushare as ts
import pandas as pd

hs300_data = ts.get_hist_data('399300')                 # 得到沪深300的日数据
values = pd.DataFrame(hs300_data['close'].to_list())    # 得到一个n行1列的dataframe
print(values.shape)                                     # 打出形状 (608,1)
df = pd.concat([values.shift(1), values], axis=1)       # 将第一列下移一格，把第二组数据按照列方式合并起来
print(df.shape)                                         # 打出形状 (608,2)
df.columns = ['t-1', 't+1']                             # 给这两列命名

result = df.corr(method='pearson')                      # pearson:  即针对线性数据的相关系数计算
                                                        # kendall:  即针对无序序列的相关系数，非正太分布的数据
                                                        # spearman: 非线性的，非正太分析的数据的相关系数
print(result)                                           # 相关性0.98　接近于1高度正相关
