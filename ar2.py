# encoding=GBK

"""
自回归模型－－AR模型的预测
"""
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR

test_size = 5

# 准备数据
hs300_data = ts.get_hist_data('399300')                 # 得到沪深300的日数据
print(hs300_data.shape)                                 # 看一下取了多少数据
s = pd.Series(hs300_data['close'].to_list())       # 得到一个series

# 划分训练／测试集
X = s.values                                               # X就是全部的数据集
train, test = X[1:len(X)-test_size], X[len(X)-test_size:]   # 从１到倒数第６个是训练集；最后５个数据作为测试集；

# 定义AR模型
model = AR(train)
# 训练AR模型
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# 使用AR模型预测test
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
# 计算模型的误差
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# 绘制实际和预测值
plt.plot(test, color='blue')                          # 用蓝色画出实际值
plt.plot(predictions, color='red')                      # 用红色画出预测值
plt.show()


# 准备数据
hs300_data = ts.get_hist_data('399300')                 # 得到沪深300的日数据
print(hs300_data.shape)                                 # 看一下取了多少数据
values = pd.DataFrame(hs300_data['close'].to_list())    # 得到一个n行1列的dataframe
df = pd.concat([values.shift(1), values], axis=1)       # 将第一列下移一格，把第二组数据按照列方式合并起来
df.columns = ['t-1', 't+1']                             # 给这两列命名

# 划分训练／测试集
X = df.values                                               # X就是全部的数据集
train, test = X[1:len(X)-test_size], X[len(X)-test_size:]   # 从１到倒数第６个是训练集；最后５个数据作为测试集；
train_X, train_y = train[:, 0], train[:, 1]                 # 训练集的ｘ是第一列；训练集的ｙ是第二列；
test_X, test_y = test[:, 0], test[:, 1]                     # 测试集的ｘ是第一列；测试集的ｙ是第二列；

# 定义持久性模型
def model_persistence(in_x):
    return in_x

# 持久性模型的预测和评估
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)    # 计算实际和预测的误差（这是 t 和 t+1 之间的误差）
print('Test MSE: %.3f' % test_score)

# 绘制实际和预测值
plt.plot(test_y, color='blue')                          # 用蓝色画出实际值
plt.plot(predictions, color='red')                      # 用红色画出预测值
plt.show()