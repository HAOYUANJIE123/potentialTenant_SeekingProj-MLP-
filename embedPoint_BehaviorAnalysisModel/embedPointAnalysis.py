import pandas as pd
import os
import numpy as np
from collections import Counter
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from DataAnalysisModel.commonCharacterAnalysis import drawBar,isNan_F
action_classifyDict_struct={
    'action_name':'',
    'code_len':0,
}


action_cdDict_struct={
    'action_name':'',
    'code_len':0,
    'parent_cd':''
}

tenantActionDict_struct={
                'label':0,
                'actions':[],
                'each_actNum':{},
                'each_actRate':{}
                         }

#行为字典构建函数
def action_cdDict_construct_F():
    action_classifyDict={}
    action_cdDict={}
    action_hiveMomDir=r'E:\PythonProject\potentialTenant_SeekingProj\Data\hiveData\action_cd_Hive'
    actionClassify_hivePath=os.path.join(action_hiveMomDir,[i for i in os.listdir(action_hiveMomDir) if 'classify' in i][0])
    action_cd_hivePath = os.path.join(action_hiveMomDir,[i for i in os.listdir(action_hiveMomDir) if 'classify' not in i][0])
    actionClassifyData=pd.read_csv(actionClassify_hivePath,dtype=str)
    action_cdData = pd.read_csv(action_cd_hivePath, dtype=str)

    #1.构建行为分类字典
    action_classify_cd_list=actionClassifyData['action_classify_cd']
    for idx in range(len(action_classify_cd_list)) :
        id=action_classify_cd_list[idx]
        action_classifyDict[id]=copy.deepcopy(action_classifyDict_struct)
        action_classifyDict[id]['action_name']=actionClassifyData['action_classify_name'][idx]
        action_classifyDict[id]['code_len'] =len(id)

    #2.构建action_cd字典
    action_cd_list = action_cdData['action_cd']
    for idx in range(len(action_cd_list)):
        id = action_cd_list[idx]
        action_cdDict[id] = copy.deepcopy(action_cdDict_struct)
        action_cdDict[id]['action_name'] = action_cdData['action_name'][idx]
        action_cdDict[id]['code_len'] = len(id)
        action_cdDict[id]['parent_cd'] = action_cdData['parent_action_cd'][idx]

    print()
    return action_classifyDict,action_cdDict

def tenantActionDict_Construct_F(need_code_len=6):
    action_classifyDict, action_cdDict=action_cdDict_construct_F()
    tenantActionDict={}
    featureMotherDir = r'E:\PythonProject\potentialTenant_SeekingProj\Data/featureData/'
    labelDictPath = os.path.join(featureMotherDir, 'labelDict.npy')
    labelDict = np.load(labelDictPath, allow_pickle=True).item()


    hiveDataMotherDir=r'E:\PythonProject\potentialTenant_SeekingProj\Data\hiveData'
    embedPointCSV=[i for i in os.listdir(hiveDataMotherDir) if i.startswith('dwd_user_embed_point_behavior_detail')][0]
    embedPointPath=os.path.join(hiveDataMotherDir,embedPointCSV)
    embedPointData=pd.read_csv(embedPointPath,dtype=str)
    embedPoint_cd_list=embedPointData['action_cd'].tolist()
    action_name_list=[]
    row_len=len(embedPoint_cd_list)
    tenant_id_list=list(labelDict.keys())

    for tenant_id in tenant_id_list:
        tenantActionDict[tenant_id] = copy.deepcopy(tenantActionDict_struct)
        tenantActionDict[tenant_id]['label']=labelDict[tenant_id]['label']

    for idx in tqdm(range(row_len)):
        tenant_id = embedPointData['tenant_id'][idx]
        if tenant_id in labelDict:
            action_cd=embedPointData['action_cd'][idx]
            ######取上级编码，取前4位是爷爷级编码，取前6位是父级编码###########

            if need_code_len==4:
                action_classify_name = action_classifyDict[action_cd[:4]]['action_name']
            else:
                action_classify_name = action_cdDict[action_cd[:6]]['action_name']

            if not (isNan_F(tenant_id) or isNan_F(action_classify_name)):
                tenantActionDict[tenant_id]['actions'].append(action_classify_name)
                if action_classify_name not in action_name_list: action_name_list.append(action_classify_name)

    # 构建 'each_actNum':{},'each_actRate':{}
    for tenant_id in tenantActionDict:

        tenantActionDict[tenant_id]['each_actNum'] = {}
        tenantActionDict[tenant_id]['each_actRate'] = {}

        for action_name in action_name_list:
            tenantActionDict[tenant_id]['each_actNum'][action_name]=0
            tenantActionDict[tenant_id]['each_actRate'][action_name] = 0.0

        #计算每个课程类数量及课程数量占比并存入字典中
        actNum_list=Counter(tenantActionDict[tenant_id]['actions']).most_common()
        if actNum_list!=[]:
            sum_actNum=sum((i[1] for i in actNum_list))
            actRate_list = [(i[0],round(i[1]/sum_actNum,3)) for i in actNum_list]
            for idx,val in enumerate(actNum_list) :
                action_name=val[0]
                act_Num=val[1]
                act_rate=actRate_list[idx][1]
                tenantActionDict[tenant_id]['each_actNum'][action_name] =act_Num
                tenantActionDict[tenant_id]['each_actRate'][action_name] = act_rate
    np.save(os.path.join(hiveDataMotherDir,'tenantActionDict.npy'),tenantActionDict)

def tenantActionAnalysis_F():
    hiveDataMotherDir = r'E:\PythonProject\potentialTenant_SeekingProj\Data\hiveData'
    tenantActionDictPath=os.path.join(hiveDataMotherDir,'tenantActionDict.npy')
    tenantActionDict=np.load(tenantActionDictPath,allow_pickle=True).item()
    #总体正式or非正式租户教学行为概览
    normal_action_list=[]
    for tenant_id in tenantActionDict:
        label=tenantActionDict[tenant_id]['label']
        if label==1:
            normal_action_list=tenantActionDict[tenant_id]['actions']
    drawBar(normal_action_list)






if __name__ == '__main__':
    tenantActionDict_Construct_F(need_code_len=6) #租户行为字典构建，need_code_len用于使用几级行为进行字典构建
    # tenantActionAnalysis_F() #非正式正式租户行为概览
