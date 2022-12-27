#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2019-08-01 00:44
公众号：AI成长社
知乎：https://www.zhihu.com/people/qlmx-61/columns
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from DeepLearningModel.train_main_github import acc_caculate_F
import os
import copy
## load data



splitMomDir = r'E:\PythonProject\potentialTenant_SeekingProj\Data\sample_dataset'
train_data = []
test_data = []

whole_data = pd.read_csv(os.path.join(splitMomDir, 'whole.csv'))
data = copy.deepcopy(whole_data)
sample_id_list = data[data.columns[0]].to_list()

cate_feature = ['tenant_level_code', 'openingtime_code', 'region_code', 'prvn_code']
for item in cate_feature:
    data[item] = LabelEncoder().fit_transform(data[item])

train = data[data['type'] == 'train']

# 对无标签数据集进行整合
predict_set_list = []
predict_set_names=['train', 'test']
for i in predict_set_names:
    if i == 'other':
        son_set = data[(data['type'] != 'train') & (data['type'] != 'test')]
        predict_set_list.append(son_set)

    elif i in ['train', 'test']:
        son_set = data[data['type'] == i]
        predict_set_list.append(son_set)

    else:
        print('子集名称错误！！！！！')

predict_set = pd.concat(predict_set_list)
##Clean up the memory
del data, train_data, test_data
gc.collect()



num_round = 500
del_feature = [train.columns[0],'virtualLabel','type']
features = [i for i in train.columns if i not in del_feature]
train_x = train[features]
train_y = train['virtualLabel'].values
test = predict_set[features]
test_y=predict_set['virtualLabel']


params = {'num_leaves': 60,
          'min_data_in_leaf': 30,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          # 'lambda_l2': 0.001,
          "verbosity": -1,
          "nthread": -1,
          'metric': {'binary_logloss', 'auc'},
          "random_state": 2019,
          # 'device': 'gpu'
          }


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
prob_oof = np.zeros((train_x.shape[0], ))
test_pred_prob = np.zeros((test.shape[0], ))

## train and predict
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x)):
    print("fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y[trn_idx])
    val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y[val_idx])


    clf = lgb.train(params,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=20,
                    categorical_feature=cate_feature,
                    early_stopping_rounds=num_round)
    prob_oof[val_idx] = clf.predict(train_x.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    test_pred_prob += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

threshold = 0.5
for pred in test_pred_prob:
    result = 1 if pred > threshold else 0
test_acc=acc_caculate_F(test_y,test_pred_prob)
print('test_acc: {}'.format(test_acc))

## plot feature importance
cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False)[:5].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',ascending=False)
plt.figure(figsize=(8, 10))
sns.barplot(y="Feature",
            x="importance",
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
# plt.savefig('../../result/lgb_importances.png')