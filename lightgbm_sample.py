# encoding=GBK

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# https://zhuanlan.zhihu.com/p/266865429
# https://jishuin.proginn.com/p/763bfbd5b842

# ��������
iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)


lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

'''
cv_results = lgb.cv(params=params,                  # ��ҪѰ�ŵĲ�����
                    train_set=lgb_train,
                    num_boost_round=1000,
                    nfold=5,
                    stratified=False,
                    shuffle=True,
                    metrics='rmse',
                    verbose_eval=50,
                    show_stdv=True,
                    seed=0
                    )

# ׼����ʼ����
cv_params = {
    'num_leaves': [5, 7, 9, 11, 13, 14, 15, 19, 23],        # һ�����ϵ�Ҷ�ӽڵ�������ٷ�˵����ҪС��2^max_depth
    'max_depth': [1, 2, 3, 4, 6, 8],                        # ��ģ�͵������ȡ���ֹ����ϵ�����Ҫ�Ĳ���
    'learning_rate': [0.05, 0.07, 0.1],                     # ѧϰ��
    'n_estimators': [100, 150, 200, 250],                   # boosting�ĵ���������һ��������ݼ�����������ѡ��100~1000֮��
    'min_child_samples': [10, 20, 30],                      # һ��Ҷ���ϵ���С������
    # 'subsample': [0.4, 0.5, 0.6, 0.7],                    # ���ݲ����ʡ����˲���С��1.0��LightGBM������ÿ�ε������ڲ������ز�������������ѡ�񲿷����ݣ�row����������������ѵ�����������ϡ�
    # 'colsample_bytree': [0.4, 0.5, 0.6, 0.7],             # ���������ʡ����˲���С��1.0��LightGBM������ÿ�ε��������ѡ�񲿷�����(col)��������������ѵ�����������ϡ�
    # 'reg_alpha': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    # 'reg_lambda': [7, 8, 9, 10],
    # 'num_iterations': [30, 40, 50],
    # 'min_data_in_leaf': [2, 10, 30, 40],
    # 'cat_smooth': [150, 160, 170, 180]
}

other_params={
    # 'max_depth': 4,
    # 'num_leaves': 15,
    # 'learning_rate': 0.07,
    # 'n_estimators': 15,
    'cat_smooth': 180,
    'subsample': 0.4,
    'colsample_bytree': 0.7,
    'reg_alpha': 3,
    'reg_lambda': 9,
}

model_lgb = lgb.LGBMRegressor(**other_params)
optimized_lgb = GridSearchCV(estimator=model_lgb,
                             param_grid=cv_params,
                             scoring='r2',
                             cv=5,
                             verbose=1,
                             n_jobs=2
                             )
optimized_lgb.fit(X_train, y_train)
print('���������ȡֵ��{0}'.format(optimized_lgb.best_params_))
print('���ģ�͵÷�:{0}'.format(optimized_lgb.best_score_))
print(optimized_lgb.cv_results_['mean_test_score'])
print(optimized_lgb.cv_results_['params'])
# ���������ȡֵ��{'learning_rate': 0.1, 'max_depth': 2, 'min_child_samples': 10, 'n_estimators': 250, 'num_leaves': 5}
'''

params = {
    'task': 'train',
    'boosting_type': 'gbdt',            # ȷ�������������ͣ�Ĭ��gbdt
    'objective': 'regression',
    'metric': {'l2', 'auc'},            #
    'max_depth': 2,
    'num_leaves': 5,
    'min_data_in_leaf': 2,              # ���������һ��ʹ��
    'min_child_samples': 10,
    'n_estimators': 250,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,            # �����������ӳ���
    'bagging_fraction': 0.8,            # ��������������������
    'bagging_freq': 5,                  # ��ζ��ÿ5�ε���ִ��bagging
    'verbose': 0                        # <0:��ʾ��������; =0:��ʾ����/����; >0:��ʾ��Ϣ
}

print("Start Training")
gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval)

print('Saving Model')
gbm.save_model('model_txt')

print('Start Predicting')

# Ԥ�����ݼ�
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# ����ģ��
print("The rmse of prediction is:{}".format(mean_squared_error(y_test, y_pred)**0.5))


