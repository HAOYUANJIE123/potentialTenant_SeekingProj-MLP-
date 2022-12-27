
import pandas as pd
import os
import numpy as np
from collections import Counter
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
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
    'teachplan_num_code': 0,  # (12)教案数量编码
    'teachplan_type_num_code': 0,  # (13)教案类型数量编码
    'teachact_num_code': 0,  # (14)教学活动数量编码
    'teachact_type_num_code': 0,  # (15)教学活动数量编码
    'openingtime_code':0,  # (16)注册时间编码，时间转整型
    'busi_region_code':0,   #(17)租户业务大区编码
    'prvn_code':0,   #(18)租户省份id编码
}

#标签字典设计
labelDict_struct={
    'label':0
}
def isNan_F(inData):
    if inData!=inData:
        return True
    else:
        return False

def reC(myDict):    # 对字典反转的第二种方法，使用压缩器
    ldict = dict(zip(myDict.values(), myDict.keys()))
    return ldict

#1.获取每个特征维度数据并进行直方图绘制
def get_X_data():
    print('开始进行特征维度直方图绘制！！！')
    startT=time.time()

    #1.读取租户特征字典以及标签字典
    featureMotherDir = r'E:\PythonProject\potentialTenant_SeekingProj\Data/featureData/'
    featureDictPath=os.path.join(featureMotherDir, 'featureDict.npy')
    labelDictPath=os.path.join(featureMotherDir, 'labelDict.npy')
    X2codeDictPath=os.path.join(featureMotherDir, 'X2codeDict.npy')

    featureDict = np.load(featureDictPath,allow_pickle=True).item()
    labelDict = np.load(labelDictPath,allow_pickle=True).item()
    X2codeDict=np.load(X2codeDictPath,allow_pickle=True).item()
    X2codeDict_reverse=reC(X2codeDict)
    tenant_id_list=list(labelDict.keys())
    first_tenant_id=tenant_id_list[0]
    x_name_list=list(featureDict[first_tenant_id].keys())

    #2.获取各x维度数据，以x1,x2保存，x1为正式租户label为1，x2为正式租户label为0

    for x_name in tqdm(x_name_list):
        x1, x2 = [], []
        for tenant_id in tenant_id_list:
            val=featureDict[tenant_id][x_name]
            if labelDict[tenant_id]['label']==1:
                x1.append(val)
            else:
                x2.append(val)
        draw_double_hist(x1,x2,x_name)
        print(x_name,x1)
    print(X2codeDict_reverse)
    endT = time.time()
    print('特征维度直方图绘制完成！！！,用时 {}s'.format(endT - startT))
def draw_double_hist(x1,x2,x_name,figureMotherDir=r'E:\PythonProject\potentialTenant_SeekingProj\Data\Figure/histFigures',isShow=True):


    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题

    sns.distplot(x1, bins=5, kde=True, hist_kws={'color': 'black'}, label='normal_Tenant')
    sns.distplot(x2, bins=100, kde=True, hist_kws={'color': 'yellow'}, label='other_Tenant')
    # plt.title('normal_other ** {}_HIST'.format(x_name))
    plt.title('正式与非正式租户特征维度 {} 比较直方图'.format(x_name))

    # plt.legend()# 显示图例
    if isShow:
        plt.show() # 显示图形
    figureNam='{}_HIST'.format(x_name)

    plt.savefig(os.path.join(figureMotherDir,figureNam))
    plt.close()

def drawBar(data):
    c=Counter(data).most_common()
    xlist=[i[0] for i in c]
    ylist = [i[1] for i in c]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题
    plt.style.use('ggplot')
    # 绘制条形图

    plt.bar(x=xlist,  # 指定条形图x轴的刻度值
    height = ylist,  # 指定条形图y轴的数值
    tick_label = xlist,  # 指定条形图x轴的刻度标签
    color = 'steelblue',  # 指定条形图的填充色
    width = 0.8
      )
    for idx, y in enumerate(ylist):
        y_rate=round(y/sum(ylist)*100,2)
        plt.text(idx, y + 0.1, '{} ，{}%'.format(round(y, 1),y_rate ) , ha='center')

    plt.show()

#2.正式租户公共特征分析
def normalTenant_commonCharacter_F():
    print('开始进行正式租户公共特征分析！！！')
    startT=time.time()

    #2.1读取租户特征字典以及标签字典
    hiveMotherDir = r'E:\PythonProject\potentialTenant_SeekingProj\Data/hiveData/'
    featureMotherDir = r'E:\PythonProject\potentialTenant_SeekingProj\Data/featureData/'

    labelDictPath = os.path.join(featureMotherDir, 'labelDict.npy')
    hive_tenantDictPath=os.path.join(hiveMotherDir, 'hive_tenantDict.npy')


    hive_tenantDict = np.load(hive_tenantDictPath, allow_pickle=True).item()
    labelDict = np.load(labelDictPath, allow_pickle=True).item()
    tenant_ids=list(labelDict.keys())
    first_tenant_id = tenant_ids[0]
    x_name_list = list(hive_tenantDict[first_tenant_id].keys())
    x_name_list=['normal_courselist','trial_courselist']
    nofreeCourse=[]
    for x_name in x_name_list:
        vals=[]
        for id in tenant_ids:
            if labelDict[id]['label']==1:
                x_name_val=hive_tenantDict[id][x_name]
                vals.append(x_name_val)
                nofreeCourse+=[i['id'] for i in x_name_val]
        if vals!=[]:
            print(x_name,vals,len(vals))
    np.save(r'E:\PythonProject\potentialTenant_SeekingProj\Data\featureData\normalTenant_nofreeCourseNam.npy',list(set(nofreeCourse)))
    drawBar(nofreeCourse)

def predict_tenant_commonCharact_F():
    print('开始进行特征维度直方图绘制！！！')
    startT=time.time()

    #1.读取租户特征字典以及标签字典
    featureMotherDir = r'E:\PythonProject\potentialTenant_SeekingProj\Data/featureData/'
    featureDictPath=os.path.join(featureMotherDir, 'featureDict.npy')
    predict_csv = r'E:\PythonProject\potentialTenant_SeekingProj\Data\predict_result\predict_detail.csv'
    predict_df = pd.read_csv(predict_csv)
    predict_df_cols=predict_df.columns.to_list()
    predict_tenant_ids = predict_df[predict_df_cols[0]].to_list()
    predict_tenant_ids=[str(i) for i in predict_tenant_ids]

    labelDictPath = os.path.join(featureMotherDir, 'labelDict.npy')
    labelDict = np.load(labelDictPath, allow_pickle=True).item()
    tenant_id_list = list(labelDict.keys())



    X2codeDictPath=os.path.join(featureMotherDir, 'X2codeDict.npy')
    featureDict = np.load(featureDictPath,allow_pickle=True).item()

    X2codeDict=np.load(X2codeDictPath,allow_pickle=True).item()
    X2codeDict_reverse=reC(X2codeDict)

    first_tenant_id=tenant_id_list[0]
    x_name_list=list(featureDict[first_tenant_id].keys())

    #2.获取各x维度数据，以x1,x2保存，x1为正式租户label为1，x2为正式租户label为0

    for x_name in tqdm(x_name_list):
        x1, x2 = [], []
        for tenant_id in tenant_id_list:
            val=featureDict[tenant_id][x_name]
            if tenant_id in predict_tenant_ids:
                x1.append(val)
            else:
                x2.append(val)
        draw_double_hist(x1,x2,x_name,figureMotherDir=r'E:\PythonProject\potentialTenant_SeekingProj\Data\Figure\predict_tenant_hist',isShow=False)
        print(x_name,x1)
    print(X2codeDict_reverse)
    endT = time.time()
    print('特征维度直方图绘制完成！！！,用时 {}s'.format(endT - startT))

def predct_with_deepTenant_analysis_F():
    tenantDictPath=r'E:\PythonProject\potentialTenant_SeekingProj\Data\hiveData\hive_tenantDict.npy'
    labelDictPath= r'E:\PythonProject\potentialTenant_SeekingProj\Data/featureData/labelDict.npy'
    predict_csv = r'E:\PythonProject\potentialTenant_SeekingProj\Data\predict_result\predict.csv'
    deep_tenantPath = r'E:\PythonProject\potentialTenant_SeekingProj\Data\predict_result\deept.csv'
    deep_tenantPath = r'E:\潜在用户分析\潜在用户预测结果\hyj授课进度50.csv'

    tenantDict=np.load(tenantDictPath, allow_pickle=True).item()
    labelDict = np.load(labelDictPath, allow_pickle=True).item()
    tenant_id_list = list(labelDict.keys())
    normal_tenant_id_list = [i for i in labelDict if labelDict[i]['label'] == 1]
    normal_tenant_name_list=[tenantDict[i]['parent_name']+tenantDict[i]['tenant_name'] for i in normal_tenant_id_list]

    tenant_name_list=[tenantDict[i]['parent_name']+tenantDict[i]['tenant_name'] for i in tenant_id_list]




    predict_df = pd.read_csv(predict_csv)
    predict_tenant_ids = predict_df[predict_df['pred']>0.5]['sample_id'].to_list()
    predict_tenant_ids = [str(i) for i in predict_tenant_ids]
    predict_tenant_names = [tenantDict[i]['parent_name']+tenantDict[i]['tenant_name'] for i in predict_tenant_ids]

    deep_tenant_df=pd.read_csv(deep_tenantPath)
    deep_T_cols=deep_tenant_df.columns.to_list()
    deep_tenant_df=deep_tenant_df[deep_tenant_df['max_plan_rate']>=100.0]
    print()
    deep_tenant_df=deep_tenant_df.assign(name=deep_tenant_df[deep_T_cols[0]]+ deep_tenant_df[deep_T_cols[1]])

    deep_names=deep_tenant_df['name'].to_list()
    useful_deep_names=[i for i in deep_names if i in tenant_name_list]
    u_deep,r_deep=get_intersection(a=predict_tenant_names,b=useful_deep_names,mom=useful_deep_names)
    u_norm, r_norm = get_intersection(a=normal_tenant_name_list, b=useful_deep_names, mom=normal_tenant_name_list)
    u_norm_deep,_=get_intersection(a=normal_tenant_name_list, b=deep_names, mom=deep_names)
    print('deep_names_num: {} ,useful_deep_names_num: {} ,normal_tenant_name_list_num: {} ,正式深度交: {}'
          .format(len(deep_names),len(useful_deep_names),len(normal_tenant_name_list),u_norm_deep))

    print(u_deep,r_deep,u_norm,r_norm)




def get_intersection(a,b,mom):
    u=[i for i in a if i in b]
    rate=round(len(u)/len(mom),4)
    print(len(u),u)
    return len(u),rate







if __name__ == '__main__':
    # get_X_data() #获取每个特征维度数据并进行直方图绘制
    # normalTenant_commonCharacter_F() #正式租户公共特征分析，主要为nofree课程
    # predict_tenant_commonCharact_F()
    predct_with_deepTenant_analysis_F()
