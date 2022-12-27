import pandas as pd
import os
import numpy as np
from collections import Counter
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats as sts
import time
from DataAnalysisModel.commonCharacterAnalysis import drawBar,isNan_F,draw_double_hist
from FeatureConstructModel.tenantFeatureConstruct import simCaculation

def virtualLabel_construct_F(postive_n,negative_n):
    print('开始进行正式租户公共特征分析！！！')
    startT=time.time()

    #1.读取租户特征字典以及标签字典
    featureMotherDir = r'E:\PythonProject\potentialTenant_SeekingProj\Data/featureData/'
    featureDictPath = os.path.join(featureMotherDir, 'featureDict.npy')
    labelDictPath = os.path.join(featureMotherDir, 'labelDict.npy')

    featureDict = np.load(featureDictPath, allow_pickle=True).item()
    labelDict = np.load(labelDictPath, allow_pickle=True).item()
    normal_tenant_id_list=[i for i in labelDict if labelDict[i]['label']==1]

    feature_df=pd.DataFrame(featureDict).T
    cols=feature_df.columns.to_list()
    all_cols=['tenant_level_code', 'student_num_code', 'techer_num_code', 'student_rate_code',
              'techer_rate_code', 'nofree_course_num_code', 'free_course_num_code', 'nofree_course_rate_code',
              'free_course_rate_code', 'nofreecourseNum_Same2Tenant', 'teachplan_num_code', 'teachplan_type_num_code',
              'teachact_num_code', 'teachact_type_num_code', 'openingtime_code', 'region_code', 'prvn_code', 'teacher_num_code',
              'teacher_rate_code', 'teachActSim']

    row_idx_list=feature_df.index.to_list()

    del_feature = ['free_course_num_code','nofree_course_num_code','nofreecourseNum_Same2Tenant','student_num_code','teachplan_num_code' ]
    # del_feature = ['free_course_num_code']


    features=['free_course_num_code','nofree_course_num_code','nofreecourseNum_Same2Tenant','student_num_code','teachact_type_num_code','teachActSim',
              'teachplan_num_code','teachplan_type_num_code']

    # features = [i for i in features if i not in del_feature]
    features = [ 'nofree_course_num_code', 'nofreecourseNum_Same2Tenant', 'student_num_code','teachact_type_num_code', 'teachActSim','teachplan_num_code', 'teachplan_type_num_code']


    # features=cols
    print(features,len(features))
    feature_df=copy.deepcopy(feature_df[features])

    # 2.计算与正式租户集合得到final相似度
    x1=[]
    x2=[]
    final_sim_list=[]
    for tenant_id in tqdm(row_idx_list):
        res = make_virtual_label_sim(normal_tenant_id_list, tenant_id, feature_df, 'mid')
        final_sim_list.append(res)

        if tenant_id in normal_tenant_id_list:
            x1.append(res[2])
            print(res)
        else:
            x2.append(res[2])


    #3.绘制标签相似度分布
    # draw_double_hist(x1,x2,'label分布')

    #4.基于final_sim进行排序，继而贴虚拟标签

    virtualLabelDict={} #0为非正式租户，1为虚拟正式租户，3为待定租户
    sort_final_sim_list=sorted(final_sim_list,key=lambda final_sim_list:final_sim_list[2], reverse=True)

    preorder_sim_tenants=[i for i in sort_final_sim_list if i[2]>0.985]
    postorder_sim_tenants = [i for i in sort_final_sim_list if i[2] <= 0.5]
    print('preorder_num: {} ;postorder_num: {}'.format(len(preorder_sim_tenants),len(postorder_sim_tenants)))

    #5.取数据集，现取前默认150作为正例，后150作为负例
    postive_ids=[i[0] for i in sort_final_sim_list[:postive_n]]
    # postive_ids = normal_tenant_id_list

    negative_ids=[i[0] for i in sort_final_sim_list[-1*negative_n:]]

    for tenant_id in row_idx_list:
        if tenant_id in postive_ids:
            virtualLabelDict[tenant_id]={'virtualLabel':1}
        elif tenant_id in negative_ids:
            virtualLabelDict[tenant_id] = {'virtualLabel': 0}
        else:
            virtualLabelDict[tenant_id] = {'virtualLabel': 3}

    np.save(os.path.join(featureMotherDir,'virtualLabelDict.npy'),virtualLabelDict)





#根据正式租户特征，设计相似度算法，确定假设标签基于相似度
def make_virtual_label_sim(normal_tenant_id_list,id,feature_df,type):
    feature=list(feature_df.loc[id])
    sim_list=[]
    for normal_id in normal_tenant_id_list:
        normal_feature = list(feature_df.loc[normal_id])
        sim=simCaculation(x=feature,y=normal_feature,functionNam='余弦相似度')
        sim_list.append(sim)
    #确定其他租户与正式租户集合相似度
    if type=='mean':
        fianl_sim=round(sts.tmean(sim_list),3)
    if type=='mid':
        fianl_sim=round(np.median(sim_list),3)
    if type=='max':
        fianl_sim=max(sim_list)
    result=[id,sim_list,fianl_sim]
    return result



if __name__ == '__main__':
    virtualLabel_construct_F(postive_n=150,negative_n=150) #根据与正式租户的集合相似度，确定虚拟标签