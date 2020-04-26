# encoding=GBK


"""
ar自回归模型－－ARMA参数的确定
"""

import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch                                         # 条件异方差模型相关的库
import statsmodels.api as sm                            # 统计相关函数的库
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy import stats

hs300_data = ts.get_hist_data('399300')
# hs300_data = ts.get_hist_data('000001')
hs300_data = hs300_data.reindex(index=hs300_data.index[::-1])       # 倒序排列一下，时间正向增长
close_data = hs300_data['close'].to_list()
change_data = np.array(hs300_data['p_change'])


# 检验收盘价
# hs300_data['close'].plot()
# plt.show()
# t = sm.tsa.stattools.adfuller(np.array(hs300_data['close']))
# print(t)


# 检验收盘价的差分
# diff = np.diff(hs300_data['close'])
# plt.plot(diff)
# plt.show()
# t = sm.tsa.stattools.adfuller(diff)
# print(t)

# 对收盘价变化进行分析。这个也就是对于收盘价的差分的比例。
# hs300_data['close'].plot()                                        # 这个明显不平稳
# plt.show()

# hs300_data['p_change'].plot()                                       # 绘制hs300日涨跌比例图
# plt.show()

t = sm.tsa.stattools.adfuller(change_data)                          # adf检验是用来检验序列是否平稳的方式 一般来说是时间序列中的一种检验方法
# t =
# (-9.403714537357603,                                                                  # 第一个是adt检验的结果，也就是t统计量的值
# 6.090385769843424e-16,                                                                # 第二个是t统计量的P值
# 6,                                                                                    # 第三个是计算过程中用到的延迟阶数
# 603,                                                                                  # 第四个是用于ADF回归和计算的观测值的个数
# {'1%': -3.441241137539733, '5%': -2.8663450276569797, '10%': -2.569328969112426},     # 第五个是配合第一个一起看的，是在99%，95%，90%置信区间下的临界的ADF检验的值。如果第一个值比第五个值小证明平稳，反正证明不平稳。
# 2031.4755668710231)
# print(t)
print("p-value: {}".format(t[1], 4))                                # p-value几乎为0小于显著水平5%，因此序列是平稳的

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(change_data, lags=20, ax=ax1)                      # 有4个阶超出了置信边界
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(change_data, lags=20, ax=ax2)                     # 有7个阶超出了置信边界。
plt.show()

# pq_list = [(0, 4), (1, 3), (2, 1), (7, 1), (3, 1), (4, 3)]                  # 最好的是(1,3) 和上面的acf,pacf有什么关系？
# for pq in pq_list:
#     arma_pq = sm.tsa.ARMA(change_data, pq).fit(disp=0)
#     print("arma_pq {} : {} {} {}".format(pq, round(arma_pq.aic, 2), round(arma_pq.bic, 2), round(arma_pq.hqic, 2)))

# arr = np.arange(64, dtype=float).reshape(8, 8)
# for p in range(0, 8):
#     for q in range(0, 8):
#         arma_pq = sm.tsa.ARMA(change_data, (p,q)).fit(disp=0)
#         arr[p, q] = round(arma_pq.aic, 3)
# print(arr.min())                                                                # 最小的aic是2080.183
# print(arr.argmin())                                                             # 对应的坐标是13,也就是[1,5], p=1, q=5
# print(arr)


order = (1, 5)
model = sm.tsa.ARMA(change_data, order).fit(disp=0)

# 计算均值方程残差
at = change_data - model.fittedvalues
at2 = np.square(at)
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(at, label='at')
plt.legend()
plt.subplot(212)
plt.plot(at2, label='at2')
plt.legend(loc=0)
plt.show()



# ######################################################################
# X = close_data
# size = int(len(X) * 0.98)
# train, test = X[0:size], X[size:len(X)]
# history = [x for x in train]
# predictions = list()
# ######################################################################
# pd.plotting.autocorrelation_plot(close_data)            # 绘制自相关系数图
# plt.show()
#
# # fit model
# model = ARIMA(pd.Series(close_data), order=(10, 1, 0))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())
# # plot residual errors
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# residuals.plot(kind='kde')
# plt.show()
# print(residuals.describe())
#
#
# for t in range(len(test)):
#     model = ARIMA(history, order=(5,1,0))
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t]
#     history.append(obs)
#     print('predicted=%f, expected=%f' % (yhat, obs))
# error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)
#
# # 绘制实际和ARIMA模型的预测值
# plt.plot(test, color='blue')
# plt.plot(predictions, color='red')
# plt.show()