#!usr/bin/env python
#-*- coding:utf-8 _*-
import copy
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import gc
import time

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from DeepLearningModel.dpModel.model import MLP

def dataset_merge_F(predict_set_names=[],exist_whole_set=True):
    splitMomDir=r'E:\PythonProject\potentialTenant_SeekingProj\Data\sample_dataset'
    train_data=[]
    test_data=[]
    if exist_whole_set:
        whole_data= pd.read_csv(os.path.join(splitMomDir,'whole.csv'))
        data = copy.deepcopy(whole_data)
    else:
        train_data = pd.read_csv(os.path.join(splitMomDir,'train.csv'))
        test_data = pd.read_csv(os.path.join(splitMomDir,'test.csv'))
        nolabel_data=pd.read_csv(os.path.join(splitMomDir,'other.csv'))
        data = pd.concat([train_data, test_data, nolabel_data])
    sample_id_list=data[data.columns[0]].to_list()

    # category feature one_hot
    cate_feature = ['tenant_level_code', 'openingtime_code', 'region_code', 'prvn_code']
    cate_feature = []

    for item in cate_feature:
        data[item] = LabelEncoder().fit_transform(data[item])
        item_dummies = pd.get_dummies(data[item])
        item_dummies.columns = [item + str(i + 1) for i in range(item_dummies.shape[1])]
        data = pd.concat([data, item_dummies], axis=1)
    data.drop(cate_feature,axis=1,inplace=True)
    train = data[data['type'] == 'train']

    #对无标签数据集进行整合
    predict_set_list=[]
    for i in predict_set_names:
        if i =='other':
            son_set=data[(data['type'] != 'train') &(data['type'] != 'test')]
            predict_set_list.append(son_set)

        elif i in ['train','test']:
            son_set=data[data['type'] == i]
            predict_set_list.append(son_set)

        else:
            print('子集名称错误！！！！！')

    predict_set =pd.concat(predict_set_list)
    ##Clean up the memory
    del data, train_data, test_data
    gc.collect()


    return train,predict_set,sample_id_list

def data_preProcessing_F(train,predict_set,has_test_label=True):


    ## get train feature
    del_feature = ['virtualLabel','type']
    features = [i for i in train.columns if i not in del_feature]

    ## Convert the label to two categories
    train_x = train[features]
    train_y = train['virtualLabel'].values

    test_x = predict_set[features]


    ## Fill missing value
    for i in train_x.columns:
        # print(i, train_x[i].isnull().sum(), test[i].isnull().sum())
        if train_x[i].isnull().sum() != 0:
            train_x[i] = train_x[i].fillna(-1)
            test_x[i] = test_x[i].fillna(-1)

    ## normalized
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_x)
    test_X = scaler.transform(test_x)

    if has_test_label:
        test_y=predict_set['virtualLabel'].values
        return train_X,train_x,train_y,test_X,test_y
    else:
        return train_X, train_x, train_y, test_X,[]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_epoch(model,train_loader,valid_loader,x_valid,loss_fn,optimizer,cfg):

    batch_size=cfg['batch_size']

    model.train()
    avg_loss = 0.
    for i, (x_batch, y_batch) in enumerate(train_loader):
        y_pred = model(x_batch)

        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # -> accumulates the gradient (by addition) for each parameter
        optimizer.step()  # -> update weights and biases
        avg_loss += loss.item() / len(train_loader)


    model.eval()
    valid_preds_fold = np.zeros((x_valid.size(0)))
    avg_val_loss = 0.
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    return avg_loss,avg_val_loss,valid_preds_fold

def train_folder_F(train_X,train_x,train_y,cfg):

    classes=cfg['classes']
    folds=cfg['folds']
    epochs=cfg['epochs']
    batch_size = cfg['batch_size']
    learning_rate=cfg['lr']


    oof_preds = np.zeros((train_X.shape[0],))
    avg_losses_f = []
    avg_val_losses_f = []
    model = MLP(train_X.shape[1], 512, classes, dropout=0.3)

    for fold_, (trn_, val_) in enumerate(folds.split(train_x)):
        print("fold {} start!!!".format(fold_ + 1))

        x_train = Variable(torch.Tensor(train_X[trn_.astype(int)]))
        y_train = Variable(torch.Tensor(train_y[trn_.astype(int), np.newaxis]))

        x_valid = Variable(torch.Tensor(train_X[val_.astype(int)]))
        y_valid = Variable(torch.Tensor(train_y[val_.astype(int), np.newaxis]))
        valid_labels=list(train_y[val_.astype(int)])
        train_labels= list(train_y[trn_.astype(int)])

        # model = MLP(train_X.shape[1], 512, classes, dropout=0.3)

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
            avg_loss, avg_val_loss,valid_preds_fold  = train_epoch(model,train_loader,valid_loader,x_valid,loss_fn,optimizer,cfg=cfg)
            valid_acc=acc_caculate_F(valid_labels,valid_preds_fold)
            elapsed_time = time.time() - start_time
            print('fold {}:   Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_acc={:.4f} \t time={:.2f}s'.format(fold_+1,epoch + 1, epochs, avg_loss,avg_val_loss,valid_acc, elapsed_time))


        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss)

        oof_preds[val_] = valid_preds_fold
        train_auc = round(roc_auc_score(train_y, oof_preds), 4)
        train_auc = acc_caculate_F(labels=train_y, predicts=oof_preds)
        print('Fold {} ending: \t All avg_train_loss={:.4f} \t avg_val_loss={:.4f} \t train_auc={:.4f}\n'.format(fold_+1,np.average(avg_losses_f), np.average(avg_val_losses_f), train_auc))

    df_model_filePATH=r'E:\PythonProject\potentialTenant_SeekingProj\Data\dp_modelFile\df_model.pt'
    # 保存
    torch.save(model.state_dict(),df_model_filePATH)
    print('模型文件保存至 {}'.format(df_model_filePATH))

    # return model
def predict_F(test_X,test_y,cfg,sample_id_list):

    #加载数据集
    batch_size = cfg['batch_size']
    x_test = np.array(test_X)
    x_test = torch.tensor(x_test, dtype=torch.float)
    if torch.cuda.is_available():
        x_test = x_test.cuda()
    test = TensorDataset(x_test)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    # 加载模型文件
    classes = cfg['classes']
    df_model_filePATH = r'E:\PythonProject\potentialTenant_SeekingProj\Data\dp_modelFile\df_model.pt'
    model =  MLP(test_X.shape[1], 512, classes, dropout=0.3)
    model.load_state_dict(torch.load(df_model_filePATH))
    model.eval()

    #开始预测
    test_preds_fold = np.zeros((len(test_X)))
    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    NN_predictions = test_preds_fold

    #如果存在标签，则进行精度计算
    if test_y!=[]:
        test_auc = round(roc_auc_score(test_y, test_preds_fold), 4)
        test_auc=acc_caculate_F(labels=test_y,predicts=test_preds_fold)
        print('\nAll test_auc={:.4f}'.format(test_auc))

    #预测标签结果
    threshold = 0.5
    result = []
    positive_samples=[]
    for idx ,pred in enumerate(NN_predictions) :
        sample_id=sample_id_list[idx]
        predict_label=1 if pred > threshold else 0

        # print(sample_id,pred,predict_label)

        if predict_label==1:positive_samples.append(sample_id)

        if test_y!=[]:
            real_label=test_y[idx]
            result.append([sample_id,real_label,predict_label,pred])
            cols=['sample_id','real_label','predict_label','pred']
        else:
            result.append([sample_id, predict_label,pred])
            cols=['sample_id','predict_label','pred']


    predict_df = pd.DataFrame(result,columns=cols)
    predict_df.to_csv(r'E:\PythonProject\potentialTenant_SeekingProj\Data\predict_result\predict.csv',encoding='utf-8')
    print('\n模型预测完毕，预测条目数量： {}, 正例数量：{}, 负例数量：{}'.format(len(result),len(positive_samples),len(result)-len(positive_samples)))

def acc_caculate_F(labels,predicts):
    real_predict_count=0
    positive_predict_count = 0
    all_positive_label_num=len([i for i in labels if i ==1])
    threshold=0.5
    for idx,label in enumerate(labels):
        predict_label = 1 if predicts[idx] > threshold else 0
        if label==predict_label:
            real_predict_count+=1
            if label==1:positive_predict_count+=1
    acc=round(real_predict_count/len(labels),4)
    posi_acc=round(positive_predict_count/all_positive_label_num,4)
    print('正例准确率：{}'.format(posi_acc))
    return acc

def get_detail_predict_tenants(topN=0):
    featureMotherDir = r'E:\PythonProject\potentialTenant_SeekingProj\Data/featureData/'
    labelDictPath = os.path.join(featureMotherDir, 'labelDict.npy')
    labelDict = np.load(labelDictPath, allow_pickle=True).item()
    normal_ids=[i for i in labelDict if labelDict[i]['label']==1]

    #获取预测为正式租户的id
    predict_csv=r'E:\PythonProject\potentialTenant_SeekingProj\Data\predict_result\predict.csv'
    predict_df=pd.read_csv(predict_csv)
    predict_tenant_df=predict_df[predict_df['predict_label']==1].loc[:,['sample_id','pred']]
    predict_tenant_df_sort=predict_tenant_df.sort_values(by='pred',ascending=False,inplace=False)

    if topN!=0:
        predict_tenant_df_sort_ids = predict_tenant_df_sort['sample_id']
        predict_tenant_df_sort_ids=predict_tenant_df_sort_ids[:topN]
    else:
        predict_tenant_df_sort_ids=predict_tenant_df_sort[predict_tenant_df_sort['pred']==1.0]['sample_id']


    for idx,id in enumerate(predict_tenant_df_sort_ids):
        single_df=predict_tenant_df[predict_tenant_df['sample_id']==id]
        print('序{}  租户id:{}, 得分：{}'.format(idx+1,id,list(single_df['pred'])))

    predict_tenant_ids=[str(i) for i in predict_tenant_df_sort_ids]

    #获取租户字典
    tenantDict_npyPath = r'E:\PythonProject\potentialTenant_SeekingProj\Data/hiveData/hive_tenantDict.npy'
    tenantDict = np.load(tenantDict_npyPath, allow_pickle=True).item()
    tenant_df=pd.DataFrame.from_dict(tenantDict).T
    tenant_df_cols=tenant_df.columns.to_list()

    #选取保存预测正式租户详细信息
    predict_tenant_df=tenant_df.loc[predict_tenant_ids,tenant_df_cols[:6]]
    isNormals=[]
    for i in predict_tenant_ids:
        res='yes' if i in normal_ids else ''
        isNormals.append(res)
    predict_tenant_df=predict_tenant_df.assign(isNormal=isNormals)



    predict_tenant_df.to_csv(r'E:\PythonProject\potentialTenant_SeekingProj\Data\predict_result\predict_detail.csv',encoding='utf-8-sig')


    #现有租户在预测租户中出现占比
    union_normals=[i for i in normal_ids if i in predict_tenant_ids]
    print('真实租户数：{}, 预测中真实租户数：{}, 交占比：{}'.format(len(normal_ids),len(union_normals),len(union_normals)/len(normal_ids)))




def wholeProject_F(just_predict=True,predict_set_names=[]):
    ##定义训练参数
    cfg={'epochs':200,
         'batch_size':20,
         'classes':1,
         'lr':0.01,
         'folds':KFold(n_splits=5, shuffle=True, random_state=2019)
         }



    ### 1.从CSV文件读取数据集并整合；predict_set_names代表你想预测的子集名称，当只存在‘test’时表示对测试集预测，list中有‘other’时has_test_label必须为False


    train_set, predict_set,sample_id_list = dataset_merge_F(predict_set_names=predict_set_names,exist_whole_set=True)


    ###2.对数据集进行预处理,包括取特征，特征标准化，空值处理，has_test_label原来说明测试集是否有标签
    has_test_label=True
    if 'other' in predict_set_names:
        has_test_label=False

    train_X, train_x, train_y, test_X, test_y = data_preProcessing_F(train=train_set, predict_set=predict_set,has_test_label=has_test_label)


    ### 3.模型训练以及预测控制
    if just_predict:
        # 3.2模型预测
        predict_F(test_X=test_X, test_y=test_y,cfg=cfg,sample_id_list=sample_id_list)
    else:
        # 3.1模型训练
        train_folder_F(train_X, train_x, train_y, cfg)
        # 3.2模型预测
        predict_F(test_X=test_X, test_y=test_y, cfg=cfg,sample_id_list=sample_id_list)






if __name__ == '__main__':
    # wholeProject_F(just_predict=True,predict_set_names=['train','test','other'])
    # wholeProject_F(just_predict=True, predict_set_names=['train', 'test'])
    get_detail_predict_tenants(topN=0) #根据预测结果二分类得分进行排序，选取topN的租户



