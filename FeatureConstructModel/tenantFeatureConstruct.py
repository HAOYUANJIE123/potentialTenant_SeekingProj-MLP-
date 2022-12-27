import pandas as pd
import os
import numpy as np
from collections import Counter
import copy
import time
from scipy import spatial
from scipy.stats import pearsonr
from embedPoint_BehaviorAnalysisModel.embedPointAnalysis import tenantActionDict_Construct_F
#租户字典结构设计
tenantDict_struct={
    'tenant_name':'',   #租户名称
    'parent_name':'',   #租户父级名称
    'tenant_level_cd':'',   #租户层次
    'opening_time':'',   #租户注册时间,取年份数据
    'region_cd':'',   #租户业务大区
    'prvn_cd':'',   #租户省份id
    'studentlist':[],   #租户学生id列表
    'teacherlist':[],   #租户教师id列表
    'normal_courselist':[],   #租户购买课程id列表
    'trial_courselist':[],   #租户试用课程id列表
    'free_courselist':[],   #租户免费课程id列表
    'teachplan_name':[],   #租户教案名称列表
    'activity_type':[],   #租户教学活动列表
}

#特征工程字典结构设计
featureDict_struct ={
    'tenant_level_code': 0,  # (1)租户层次特征编码
    'student_num_code': 0,  # (2)学生数量编码
    'techer_num_code': 0,  # (3)教师数量编码
    'student_rate_code': 0,  # (5)学生增长率编码，取整
    'techer_rate_code': 0,  # (6)教师增长率编码，取整
    'nofree_course_num_code': 0,  # (8)非免费课程数量编码，即试用、购买课程
    'free_course_num_code': 0,  # (9)免费课程数量编码
    'nofree_course_rate_code': 0,  # (10)非免费课程增长率编码，取整
    'free_course_rate_code': 0,  # (11)免费课程增长率编码，取整
    'nofreecourseNum_Same2Tenant':0, #(19)与正式租户nofree课程相同数量
    'teachplan_num_code': 0,  # (12)教案数量编码
    'teachplan_type_num_code': 0,  # (13)教案类型数量编码
    'teachact_num_code': 0,  # (14)教学活动数量编码
    'teachact_type_num_code': 0,  # (15)教学活动数量编码
    'openingtime_code':0,  # (16)注册时间编码，时间转整型
    'region_code':0,   #(17)租户业务大区编码
    'prvn_code':0,   #(18)租户省份id编码

}

#标签字典设计
labelDict_struct={
    'label':0
}


#对字符串进行ascll编码
def string2Ascll_F(string=''):
    ascllList=[str(ord(i)) for i in string]
    result=''.join(ascllList)
    return int(result)

def isNan_F(inData):
    if inData!=inData:
        return True
    else:
        return False

#计算增长率、数量
def growthRateCacultion_F(in_set=[]):
    '''
    :param in_set: [{'id':'','year':''},...]
    :return:growthrate,num
    '''
    num=len(in_set)
    years=[i['year'] for i in in_set]
    c=Counter(years).most_common()
    growthrate=0
    if len(c)>1:
        sort_c=sorted(c,key = lambda c : c[0], reverse = True)
        growthrate =((sort_c[0][1])-(sort_c[1][1]))//(sort_c[1][1])

    return growthrate,num

#对原生信息进行特征编码
def X_to_code_F():
    X2codeDict={}

    tenantDict_npyPath=r'E:\PythonProject\potentialTenant_SeekingProj\Data/hiveData/hive_tenantDict.npy'
    tenantDict = np.load(tenantDict_npyPath,allow_pickle=True).item()
    tenant_id_list=tenantDict.keys()
    wrongCount=0
    for tenant_id in tenant_id_list:
        tenant_level_cd=tenantDict[tenant_id]['tenant_level_cd']
        region_cd=tenantDict[tenant_id]['region_cd']
        prvn_cd=tenantDict[tenant_id]['prvn_cd']
        if tenant_level_cd!=tenant_level_cd or region_cd!=region_cd or prvn_cd!=prvn_cd:
            wrongCount+=1

        if tenant_level_cd==tenant_level_cd and tenant_level_cd not in X2codeDict:
            # 开始进行字符串编码，编码原则：字符ascll编码拼接并转int
            X2codeDict[tenant_level_cd]=string2Ascll_F(tenant_level_cd)

        if region_cd==region_cd and region_cd not in X2codeDict:
            X2codeDict[region_cd]=string2Ascll_F(region_cd)

        if prvn_cd==prvn_cd and prvn_cd not in X2codeDict:
            X2codeDict[prvn_cd]=int(prvn_cd)

    return tenantDict,X2codeDict,tenant_id_list

#租户特征构建
def featureDictConstruct_F():
    print('开始进行特征字典构建！！！')
    startT=time.time()
    featureMotherDir=r'E:\PythonProject\potentialTenant_SeekingProj\Data\featureData'
    normalTenant_nofreeCourseNamPath=r'E:\PythonProject\potentialTenant_SeekingProj\Data\featureData\normalTenant_nofreeCourseNam.npy'
    normalTenant_nofreeCourseNam=np.load(normalTenant_nofreeCourseNamPath).tolist()
    tenantDict, X2codeDict,tenant_id_list=X_to_code_F()
    featureDict={}
    labelDict={}

    for id in tenant_id_list:
        #1.读取租户字典数据，并进行对应指标计算
        tenant_level=tenantDict[id]['tenant_level_cd']

        studentlist=tenantDict[id]['studentlist']
        studentRate,studentNum=growthRateCacultion_F(studentlist)#计算数量和增长率

        teacherlist = tenantDict[id]['teacherlist']
        teacherRate, teacherNum = growthRateCacultion_F(teacherlist)

        nofree_courseList=tenantDict[id]['normal_courselist']+tenantDict[id]['trial_courselist']
        nofree_courseRate, nofree_courseNum = growthRateCacultion_F(nofree_courseList)

        free_courseList = tenantDict[id]['free_courselist']
        free_courseRate, free_courseNum = growthRateCacultion_F(free_courseList)

        teachplanList=tenantDict[id]['teachplan_name']
        teachactList = tenantDict[id]['activity_type']

        region=tenantDict[id]['region_cd']
        prvn_cd=tenantDict[id]['prvn_cd']

        #2.保存非Nan数据的租户并构建租户特征
        if not(isNan_F(tenant_level) or isNan_F(region) or isNan_F(prvn_cd)):

            featureDict[id]=copy.deepcopy(featureDict_struct)
            labelDict[id]=copy.deepcopy(labelDict_struct)
            normal_courselist=tenantDict[id]['normal_courselist']
            if len(normal_courselist)>0:
                labelDict[id]['label']=1

            # 3.对特征字典进行赋值
            featureDict[id]['tenant_level_code']=X2codeDict[tenant_level]
            featureDict[id]['student_num_code'] = studentNum
            featureDict[id]['student_rate_code'] = studentRate
            featureDict[id]['teacher_num_code'] = teacherNum
            featureDict[id]['teacher_rate_code'] = teacherRate
            featureDict[id]['nofree_course_num_code'] =nofree_courseNum
            featureDict[id]['free_course_num_code'] =free_courseNum
            featureDict[id]['nofree_course_rate_code'] =nofree_courseRate
            featureDict[id]['free_course_rate_code'] = free_courseRate
            featureDict[id]['nofreecourseNum_Same2Tenant'] = len([i['id'] for i in nofree_courseList if i['id'] in normalTenant_nofreeCourseNam] )
            featureDict[id]['teachplan_num_code'] =len(teachplanList)
            featureDict[id]['teachplan_type_num_code'] =len(list(set(teachplanList)))
            featureDict[id]['teachact_num_code'] =len(teachactList)
            featureDict[id]['teachact_type_num_code'] =len(list(set(teachactList)))
            featureDict[id]['openingtime_code'] =int(tenantDict[id]['opening_time'])
            featureDict[id]['region_code'] =X2codeDict[region]
            featureDict[id]['prvn_code'] =X2codeDict[prvn_cd]

            #如果要加入租户行为维度则执行下面，#相似度*100取整
            featureDict[id]['teachActSim']=int(get_teachActSim_F(id)*100)

    np.save(os.path.join(featureMotherDir,'featureDict.npy'),featureDict)
    np.save(os.path.join(featureMotherDir, 'labelDict.npy'), labelDict)
    np.save(os.path.join(featureMotherDir, 'X2codeDict.npy'), X2codeDict)
    endT = time.time()
    print('特征字典构建完成！！！,用时 {}s'.format(endT - startT))

def get_teachActSim_F(tenant_id):

    hiveDataMotherDir = r'E:\PythonProject\potentialTenant_SeekingProj\Data\hiveData'
    tenantActionDictPath = os.path.join(hiveDataMotherDir, 'tenantActionDict.npy')
    tenantActionDict = np.load(tenantActionDictPath, allow_pickle=True).item()
    label=tenantActionDict[tenant_id]['label']

    normal_actRate = [0.569, 0.106, 0.097, 0.0568, 0.0555, 0.0537, 0.0467, 0.0138, 0.0014]
    # normal_actRate =[0.68,0.21,0.06,0.05]
    simVal=0.0
    # if label==1:
    actRate_list=list(tenantActionDict[tenant_id]['each_actRate'].items())
    actRate_val_list=[i[1] for i in actRate_list]
    simVal=simCaculation(x=actRate_val_list,y=normal_actRate,functionNam='余弦相似度')
    if isNan_F(simVal):
        simVal=0.0
    return simVal


def simCaculation(x,y,functionNam):
    '''
    
    :param x: list
    :param y: list
    :param functionNam: str
    :return: simVal
    '''
    x=np.array(x)
    y=np.array(y)
    simVal=0.0
    if functionNam=='欧式相似度':
        simVal=np.sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))

    if functionNam=='余弦相似度':
        simVal=1 - spatial.distance.cosine(x, y)

    if functionNam=='皮尔逊相似度':
        simVal=pearsonr(x,y)[0]

    if functionNam == '，曼哈顿相似度':
        simVal = sum(abs(a-b) for a,b in zip(x,y))
    simVal=round(simVal,2)
    return simVal

if __name__ == '__main__':
    # X_to_code_F()  #对字符串进行编码，构建映射
    featureDictConstruct_F()  #特征字典构建