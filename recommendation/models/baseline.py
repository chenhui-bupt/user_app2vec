# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_auc_score
from sklearn.model_selection import train_test_split


data = pd.read_csv('../data/deepwalk.csv')
data.drop('app_cat', inplace=True)
X = data.iloc[:, 4:].values
y = data['click'].values
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 32,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=50,
                    )
f1_score(y_test, np.where(gbm.predict(X_test, num_iteration=gbm.best_iteration) > 0.5, 1, 0))
