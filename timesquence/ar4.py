# encoding=GBK


"""
ar�Իع�ģ�ͣ���ARMA������ȷ��
"""

import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch                                         # �����췽��ģ����صĿ�
import statsmodels.api as sm                            # ͳ����غ����Ŀ�
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy import stats

hs300_data = ts.get_hist_data('399300')
# hs300_data = ts.get_hist_data('000001')
hs300_data = hs300_data.reindex(index=hs300_data.index[::-1])       # ��������һ�£�ʱ����������
close_data = hs300_data['close'].to_list()
change_data = np.array(hs300_data['p_change'])


# �������̼�
# hs300_data['close'].plot()
# plt.show()
# t = sm.tsa.stattools.adfuller(np.array(hs300_data['close']))
# print(t)


# �������̼۵Ĳ��
# diff = np.diff(hs300_data['close'])
# plt.plot(diff)
# plt.show()
# t = sm.tsa.stattools.adfuller(diff)
# print(t)

# �����̼۱仯���з��������Ҳ���Ƕ������̼۵Ĳ�ֵı�����
# hs300_data['close'].plot()                                        # ������Բ�ƽ��
# plt.show()

# hs300_data['p_change'].plot()                                       # ����hs300���ǵ�����ͼ
# plt.show()

t = sm.tsa.stattools.adfuller(change_data)                          # adf�������������������Ƿ�ƽ�ȵķ�ʽ һ����˵��ʱ�������е�һ�ּ��鷽��
# t =
# (-9.403714537357603,                                                                  # ��һ����adt����Ľ����Ҳ����tͳ������ֵ
# 6.090385769843424e-16,                                                                # �ڶ�����tͳ������Pֵ
# 6,                                                                                    # �������Ǽ���������õ����ӳٽ���
# 603,                                                                                  # ���ĸ�������ADF�ع�ͼ���Ĺ۲�ֵ�ĸ���
# {'1%': -3.441241137539733, '5%': -2.8663450276569797, '10%': -2.569328969112426},     # ���������ϵ�һ��һ�𿴵ģ�����99%��95%��90%���������µ��ٽ��ADF�����ֵ�������һ��ֵ�ȵ����ֵС֤��ƽ�ȣ�����֤����ƽ�ȡ�
# 2031.4755668710231)
# print(t)
print("p-value: {}".format(t[1], 4))                                # p-value����Ϊ0С������ˮƽ5%�����������ƽ�ȵ�

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(change_data, lags=20, ax=ax1)                      # ��4���׳��������ű߽�
ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(change_data, lags=20, ax=ax2)                     # ��7���׳��������ű߽硣
plt.show()

# pq_list = [(0, 4), (1, 3), (2, 1), (7, 1), (3, 1), (4, 3)]                  # ��õ���(1,3) �������acf,pacf��ʲô��ϵ��
# for pq in pq_list:
#     arma_pq = sm.tsa.ARMA(change_data, pq).fit(disp=0)
#     print("arma_pq {} : {} {} {}".format(pq, round(arma_pq.aic, 2), round(arma_pq.bic, 2), round(arma_pq.hqic, 2)))

# arr = np.arange(64, dtype=float).reshape(8, 8)
# for p in range(0, 8):
#     for q in range(0, 8):
#         arma_pq = sm.tsa.ARMA(change_data, (p,q)).fit(disp=0)
#         arr[p, q] = round(arma_pq.aic, 3)
# print(arr.min())                                                                # ��С��aic��2080.183
# print(arr.argmin())                                                             # ��Ӧ��������13,Ҳ����[1,5], p=1, q=5
# print(arr)


order = (1, 5)
model = sm.tsa.ARMA(change_data, order).fit(disp=0)

# �����ֵ���̲в�
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
# pd.plotting.autocorrelation_plot(close_data)            # ���������ϵ��ͼ
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
# # ����ʵ�ʺ�ARIMAģ�͵�Ԥ��ֵ
# plt.plot(test, color='blue')
# plt.plot(predictions, color='red')
# plt.show()