# -*- coding: utf-8 -*-

import tushare as ts
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

colors = ['blue', 'red', 'green', 'black', 'orange', 'yellow']

data = ts.get_hist_data('600000', start='2019-01-08')

print(data.columns)

open = data['open'].values
high = data['high'].values
low = data['low'].values
close = data['close'].values
volume = data['volume'].values
ma5 = data['ma5'].values
ma10 = data['ma10'].values
ma20 = data['ma20'].values
vma5 = data['v_ma5'].values
vma10 = data['v_ma10'].values
vma20 = data['v_ma20'].values

dates = np.array([i for i in range(data.shape[0])])

diff = np.diff(close)

dates = dates[1:]
open = open[1:]
high = high[1:]
close = close[1:]
low = low[1:]
volume = volume[1:]
ma5 = ma5[1:]
ma10 = ma10[1:]
ma20 = ma20[1:]
vma5 = vma5[1:]
vma10 = vma10[1:]
vma20 = vma20[1:]

X = np.column_stack([diff, open, high, close, low, volume, ma5, ma10, ma20, vma5, vma10, vma20])

print("观测值：")
print(X)

diff_v = diff.reshape(-1, 1)


n = 4

model = GaussianHMM(n_components=n, n_iter=1000, covariance_type='full', tol=0.0001)

model = model.fit(X)

print("样本量：")
print(X.shape)
print("给定的隐藏特征数目：")
print(n)
print("初始的隐藏状态概率π：")
print(model.startprob_)
print("状态转移矩阵A参数：")
print(model.transmat_)
print("估计均值：")
print(model.means_)
print("估计方差：")
print(model.covars_)
print("预测的概率：")
y = model.predict_proba(X)
print(y)
hidden_states = model.predict(X)
print("预测状态值：")
print(hidden_states)
print(model.score(X))

# HMM模型只是能分离出不同的状态，具体对每个状态赋予现实的市场意义，是需要人为来辨别和观察的。
for j in range(len(close)-1):
    for i in range(model.n_components):
        if hidden_states[j] == i:
            plt.plot([dates[j], dates[j+1]], [close[j], close[j+1]], color=colors[i])
plt.show()

# import pandas as pd
# # data = pd.DataFrame({'datelist': dates, 'close': close, 'state': hidden_states}).set_index('dates')
# data = pd.DataFrame({'datelist': dates, 'close': close, 'state': hidden_states})
# plt.figure(figsize=(15,8))
# for i in range(model.n_components):
#     state = (hidden_states == i)
#     idx = np.append(0, state[:-1])
#     data['state %d_return'%i] = data.close.multiply(idx, axis = 0)
#     plt.plot(np.exp(data['state %d_return' %i].cumsum()),label = 'hidden_states %d'%i)
#     plt.legend()
#     plt.grid(1)

# real_states = list(apply(lambda x: 1 if x>0 else 0), diff_v)