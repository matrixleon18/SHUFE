# encoding=GBK

"""
�Իع�ģ�ͣ���ARģ�͵�Ԥ��
"""
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR

test_size = 5

# ׼������
hs300_data = ts.get_hist_data('399300')                 # �õ�����300��������
print(hs300_data.shape)                                 # ��һ��ȡ�˶�������
s = pd.Series(hs300_data['close'].to_list())       # �õ�һ��series

# ����ѵ�������Լ�
X = s.values                                               # X����ȫ�������ݼ�
train, test = X[1:len(X)-test_size], X[len(X)-test_size:]   # �ӣ��������ڣ�����ѵ��������󣵸�������Ϊ���Լ���

# ����ARģ��
model = AR(train)
# ѵ��ARģ��
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# ʹ��ARģ��Ԥ��test
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
# ����ģ�͵����
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# ����ʵ�ʺ�Ԥ��ֵ
plt.plot(test, color='blue')                          # ����ɫ����ʵ��ֵ
plt.plot(predictions, color='red')                      # �ú�ɫ����Ԥ��ֵ
plt.show()


# ׼������
hs300_data = ts.get_hist_data('399300')                 # �õ�����300��������
print(hs300_data.shape)                                 # ��һ��ȡ�˶�������
values = pd.DataFrame(hs300_data['close'].to_list())    # �õ�һ��n��1�е�dataframe
df = pd.concat([values.shift(1), values], axis=1)       # ����һ������һ�񣬰ѵڶ������ݰ����з�ʽ�ϲ�����
df.columns = ['t-1', 't+1']                             # ������������

# ����ѵ�������Լ�
X = df.values                                               # X����ȫ�������ݼ�
train, test = X[1:len(X)-test_size], X[len(X)-test_size:]   # �ӣ��������ڣ�����ѵ��������󣵸�������Ϊ���Լ���
train_X, train_y = train[:, 0], train[:, 1]                 # ѵ�����ģ��ǵ�һ�У�ѵ�����ģ��ǵڶ��У�
test_X, test_y = test[:, 0], test[:, 1]                     # ���Լ��ģ��ǵ�һ�У����Լ��ģ��ǵڶ��У�

# ����־���ģ��
def model_persistence(in_x):
    return in_x

# �־���ģ�͵�Ԥ�������
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)    # ����ʵ�ʺ�Ԥ��������� t �� t+1 ֮�����
print('Test MSE: %.3f' % test_score)

# ����ʵ�ʺ�Ԥ��ֵ
plt.plot(test_y, color='blue')                          # ����ɫ����ʵ��ֵ
plt.plot(predictions, color='red')                      # �ú�ɫ����Ԥ��ֵ
plt.show()