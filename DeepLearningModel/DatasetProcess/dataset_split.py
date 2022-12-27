import pandas as pd
import os
import numpy as np
import random
from collections import Counter
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
def datasetSplit_F0():
    featureMomDir=r'E:\PythonProject\potentialTenant_SeekingProj\Data\featureData'
    datasetPath=r'E:\PythonProject\potentialTenant_SeekingProj\Data\sample_dataset'
    virtualLabelPath=os.path.join(featureMomDir,'virtualLabelDict.npy')
    virtualLabelDict=np.load(virtualLabelPath,allow_pickle=True).item()
    virtualLabel_df=pd.DataFrame(virtualLabelDict).T
    row_ids=virtualLabel_df.index.to_list()
    dataset_ids=[id for id in row_ids if list(virtualLabel_df.loc[id])[0]<3]
    random.shuffle(dataset_ids)
    train_num=int(len(dataset_ids)*0.7)
    train_id_list=dataset_ids[:train_num]
    test_id_list=dataset_ids[train_num:]
    datasetSplitDict={'train_dataset':train_id_list,'test_dataset':test_id_list}
    np.save(os.path.join(datasetPath,'datasetSplitDict.npy'),datasetSplitDict)
    print('训练集数量：{}, 测试集数量：{}'.format(len(train_id_list),len(test_id_list)))

def datasetSplit_F():

    featureMomDir = r'E:\PythonProject\potentialTenant_SeekingProj\Data\featureData'
    datasetPath = r'E:\PythonProject\potentialTenant_SeekingProj\Data\sample_dataset'
    virtualLabelPath = os.path.join(featureMomDir, 'virtualLabelDict.npy')
    featurePath=os.path.join(featureMomDir,'featureDict.npy')
    virtualLabelDict = np.load(virtualLabelPath, allow_pickle=True).item()
    featureDict = np.load(featurePath, allow_pickle=True).item()
    virtualLabel_df = pd.DataFrame(virtualLabelDict).T
    featureDict_df = pd.DataFrame(featureDict).T
    final_df=pd.concat([featureDict_df,virtualLabel_df],axis=1)

    row_ids = virtualLabel_df.index.to_list()
    dataset_ids = [id for id in row_ids if list(virtualLabel_df.loc[id])[0] < 3]
    random.shuffle(dataset_ids)
    train_num = int(len(dataset_ids) * 0.7)

    train_id_list = dataset_ids[:train_num]

    test_id_list = dataset_ids[train_num:]
    other_id_list=[id for id in row_ids if list(virtualLabel_df.loc[id])[0] == 3]

    train_df=final_df.loc[train_id_list]
    test_df=final_df.loc[test_id_list]
    other_df=final_df.loc[other_id_list]
    train_df=train_df.assign(type='train')
    test_df=test_df.assign(type='test')
    other_df=other_df.assign(type='other')

    whole_df=pd.concat([train_df,test_df,other_df])


    train_df.to_csv(os.path.join(datasetPath,'train.csv'))
    test_df.to_csv(os.path.join(datasetPath, 'test.csv'))
    other_df.to_csv(os.path.join(datasetPath, 'other.csv'))
    whole_df.to_csv(os.path.join(datasetPath, 'whole.csv'))

    datasetSplitDict = {'train_dataset': train_id_list, 'test_dataset': test_id_list,'other_dataset': other_id_list}
    np.save(os.path.join(datasetPath, 'datasetSplitDict.npy'), datasetSplitDict)

    print('训练集数量：{}, 测试集数量：{}, 剩余集数量：{} '.format(len(train_id_list), len(test_id_list),len(other_id_list)))




if __name__ == '__main__':
    datasetSplit_F()
