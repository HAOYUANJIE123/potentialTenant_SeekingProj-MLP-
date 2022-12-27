#!usr/bin/env python
#-*- coding:utf-8 _*-
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import gc
import time

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from dpModel.model import MLP


## load data
splitMomDir=r'E:\PythonProject\potentialTenant_SeekingProj\Data\sample_dataset'
train_data = pd.read_csv(os.path.join(splitMomDir,'train.csv'))
test_data = pd.read_csv(os.path.join(splitMomDir,'test.csv'))
train_id_list=train_data.index.to_list()
test_id_list=test_data.index.to_list()
epochs = 50
batch_size = 10
classes = 1
learning_rate = 0.01


## category feature one_hot
data = pd.concat([train_data, test_data])
cate_feature = ['tenant_level_code', 'openingtime_code', 'region_code', 'prvn_code']
for item in cate_feature:
    data[item] = LabelEncoder().fit_transform(data[item])
    item_dummies = pd.get_dummies(data[item])
    item_dummies.columns = [item + str(i + 1) for i in range(item_dummies.shape[1])]
    data = pd.concat([data, item_dummies], axis=1)
data.drop(cate_feature,axis=1,inplace=True)

# train=data[data[cols[0]] in train_id_list]
train=data[data['type'] =='train']
test = data[data['type'] =='test']

##Clean up the memory
del data, train_data, test_data
gc.collect()

## get train feature
del_feature = ['virtualLabel','type']
features = [i for i in train.columns if i not in del_feature]


## Convert the label to two categories
train_x = train[features]
train_y = train['virtualLabel'].values
test_y=test['virtualLabel'].values
test = test[features]


## Fill missing value
for i in train_x.columns:
    # print(i, train_x[i].isnull().sum(), test[i].isnull().sum())
    if train_x[i].isnull().sum() != 0:
        train_x[i] = train_x[i].fillna(-1)
        test[i] = test[i].fillna(-1)

## normalized
scaler = StandardScaler()
train_X = scaler.fit_transform(train_x)
test_X = scaler.transform(test)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
NN_predictions = np.zeros((test_X.shape[0], ))
oof_preds = np.zeros((train_X.shape[0], ))

x_test = np.array(test_X)
x_test = torch.tensor(x_test, dtype=torch.float)
if torch.cuda.is_available():
    x_test = x_test.cuda()
test = TensorDataset(x_test)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

avg_losses_f = []
avg_val_losses_f = []

for fold_, (trn_, val_) in enumerate(folds.split(train_x)):
    print("fold {}".format(fold_ + 1))

    x_train = Variable(torch.Tensor(train_X[trn_.astype(int)]))
    y_train = Variable(torch.Tensor(train_y[trn_.astype(int), np.newaxis]))

    x_valid = Variable(torch.Tensor(train_X[val_.astype(int)]))
    y_valid = Variable(torch.Tensor(train_y[val_.astype(int), np.newaxis]))

    model = MLP(x_train.shape[1], 512, classes, dropout=0.3)

    if torch.cuda.is_available():
        x_train, y_train = x_train.cuda(), y_train.cuda()
        x_valid, y_valid = x_valid.cuda(), y_valid.cuda()
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()  # Combined with the sigmoid

    train = TensorDataset(x_train, y_train)
    valid = TensorDataset(x_valid, y_valid)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for i, (x_batch, y_batch) in enumerate(train_loader):
            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()       # clear gradients for next train
            loss.backward()             # -> accumulates the gradient (by addition) for each parameter
            optimizer.step()            # -> update weights and biases
            avg_loss += loss.item() / len(train_loader)
            # avg_auc += round(roc_auc_score(y_batch.cpu(),y_pred.detach().cpu()),4) / len(train_loader)
        model.eval()

        valid_preds_fold = np.zeros((x_valid.size(0)))
        test_preds_fold = np.zeros((len(test_X)))

        avg_val_loss = 0.
        # avg_val_auc = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()

            # avg_val_auc += round(roc_auc_score(y_batch.cpu(),sigmoid(y_pred.cpu().numpy())[:, 0]),4) / len(valid_loader)
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, epochs, avg_loss, avg_val_loss, elapsed_time))

    avg_losses_f.append(avg_loss)
    avg_val_losses_f.append(avg_val_loss)

    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    oof_preds[val_] = valid_preds_fold
    NN_predictions += test_preds_fold / folds.n_splits

train_auc = round(roc_auc_score(train_y, oof_preds), 4)
test_auc = round(roc_auc_score(test_y, test_preds_fold), 4)
print('All \t loss={:.4f} \t val_loss={:.4f} \t train_auc={:.4f}\t test_auc={:.4f}'.format(np.average(avg_losses_f), np.average(avg_val_losses_f), train_auc,test_auc))

threshold = 0.5
result = []
for pred in NN_predictions:
    predict_label=1 if pred > threshold else 0
    result.append(predict_label)