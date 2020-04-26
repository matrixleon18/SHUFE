# encoding=GBK


"""
ar自回归模型－－ARIMA
"""
import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

hs300_data = ts.get_hist_data('399300')
close_data = hs300_data['close'].to_list()
######################################################################
X = close_data
size = int(len(X) * 0.98)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
######################################################################
pd.plotting.autocorrelation_plot(close_data)            # 用pandas绘制自相关系数图
plt.show()

plot_acf(hs300_data['close'])            # 用sm绘制自相关系数图。横坐标表示延迟阶数，纵坐标表示自相关系数。
plt.show()

plot_pacf(hs300_data['close'])            # 用sm绘制偏自相关系数图。横坐标表示延迟阶数，纵坐标表示自相关系数。
plt.show()

# fit model
model = ARIMA(pd.Series(close_data), order=(9, 1, 0))   # d=1 就是做一阶差分。（其实应该是看一阶差分后是不是平稳再定）
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# 绘制实际和ARIMA模型的预测值
plt.plot(test, color='blue')
plt.plot(predictions, color='red')
plt.show()