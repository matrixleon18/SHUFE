# encoding=GBK

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# https://zhuanlan.zhihu.com/p/266865429
# https://jishuin.proginn.com/p/763bfbd5b842

# 载入数据
iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)


lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

'''
cv_results = lgb.cv(params=params,                  # 需要寻优的参数集
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

# 准备开始调参
cv_params = {
    'num_leaves': [5, 7, 9, 11, 13, 14, 15, 19, 23],        # 一棵树上的叶子节点个数。官方说法是要小于2^max_depth
    'max_depth': [1, 2, 3, 4, 6, 8],                        # 树模型的最大深度。防止过拟合的最重要的参数
    'learning_rate': [0.05, 0.07, 0.1],                     # 学习率
    'n_estimators': [100, 150, 200, 250],                   # boosting的迭代次数。一般根据数据集和特征数据选择100~1000之间
    'min_child_samples': [10, 20, 30],                      # 一个叶子上的最小数据量
    # 'subsample': [0.4, 0.5, 0.6, 0.7],                    # 数据采样率。若此参数小于1.0，LightGBM将会在每次迭代中在不进行重采样的情况下随机选择部分数据（row），可以用来加速训练及处理过拟合。
    # 'colsample_bytree': [0.4, 0.5, 0.6, 0.7],             # 特征采样率。若此参数小于1.0，LightGBM将会在每次迭代中随机选择部分特征(col)，可以用来加速训练及处理过拟合。
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
print('参数的最佳取值：{0}'.format(optimized_lgb.best_params_))
print('最佳模型得分:{0}'.format(optimized_lgb.best_score_))
print(optimized_lgb.cv_results_['mean_test_score'])
print(optimized_lgb.cv_results_['params'])
# 参数的最佳取值：{'learning_rate': 0.1, 'max_depth': 2, 'min_child_samples': 10, 'n_estimators': 250, 'num_leaves': 5}
'''

params = {
    'task': 'train',
    'boosting_type': 'gbdt',            # 确定估计器的类型，默认gbdt
    'objective': 'regression',
    'metric': {'l2', 'auc'},            #
    'max_depth': 2,
    'num_leaves': 5,
    'min_data_in_leaf': 2,              # 和上面参数一起使用
    'min_child_samples': 10,
    'n_estimators': 250,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,            # 决定特征的子抽样
    'bagging_fraction': 0.8,            # 建立树的样本采样比例
    'bagging_freq': 5,                  # 意味着每5次迭代执行bagging
    'verbose': 0                        # <0:显示致命错误; =0:显示错误/警告; >0:显示信息
}

print("Start Training")
gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval)

print('Saving Model')
gbm.save_model('model_txt')

print('Start Predicting')

# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# 评估模型
print("The rmse of prediction is:{}".format(mean_squared_error(y_test, y_pred)**0.5))


