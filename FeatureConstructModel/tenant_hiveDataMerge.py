import pandas as pd
import os
import numpy as np
import copy
import time
##tenantDict存储hive表得到的原生数据
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
def isNan_F(inData):
    if inData!=inData:
        return True
    else:
        return False
#构建租户字典函数
def tenantDict_Construct_F():
    print('开始进行租户字典构建！！！')
    startT=time.time()

    tenantDict = {}
    hive_motherDir='E:\PythonProject\potentialTenant_SeekingProj\Data\hiveData'
    hivetabFile=os.listdir(hive_motherDir)
    hivetab_names=['dwd_tenant_base_info','dwd_tenant_geo_location','dwd_tenant_user','dwd_tenant_course','dwd_course_teach_plan_detail','dwd_course_teach_activity_detail']

    ####1.获取表dwd_tenant_base_info中相关信息
    goal_hivetabPath = ''

    ##1.1获取目标hive表文件路径
    for file in hivetabFile:
        if 'dwd_tenant_base_info' in file:
            goal_hivetabPath=os.path.join(hive_motherDir,file)
            break

    ##1.2读取目标hive表数据
    #1.2.1 租户结构空字典定义
    goal_hivetabData = pd.read_csv(goal_hivetabPath,dtype=str)
    tenant_id_list=goal_hivetabData['tenant_id'].tolist()
    for id in tenant_id_list:
        tenantDict[id]=copy.deepcopy(tenantDict_struct)  #深拷贝字典结构

    # 1.3 租户相关字典赋值
    for idx in range(len(goal_hivetabData['tenant_id'])):
        tenant_id_val=goal_hivetabData['tenant_id'][idx]
        tenantDict[tenant_id_val]['tenant_name']=goal_hivetabData['tenant_name'][idx]
        tenantDict[tenant_id_val]['parent_name'] = goal_hivetabData['parent_name'][idx]
        tenantDict[tenant_id_val]['tenant_level_cd'] = goal_hivetabData['tenant_level_cd'][idx]
        tenantDict[tenant_id_val]['opening_time'] = goal_hivetabData['opening_time'][idx][:4]

    ####2.获取表dwd_tenant_geo_location中相关信息

    ##2.1获取目标hive表文件路径
    for file in hivetabFile:
        if 'dwd_tenant_geo_location' in file:
            goal_hivetabPath = os.path.join(hive_motherDir, file)
            break

    ##2.2读取目标hive表数据
    goal_hivetabData = pd.read_csv(goal_hivetabPath,dtype=str)

    # 2.3 租户相关字典赋值
    for idx in range(len(goal_hivetabData['tenant_id'])):
        tenant_id_val = goal_hivetabData['tenant_id'][idx]
        tenantDict[tenant_id_val]['region_cd'] = goal_hivetabData['region_cd'][idx]
        tenantDict[tenant_id_val]['prvn_cd'] = goal_hivetabData['prvn_cd'][idx]


    ####3.获取表dwd_tenant_user中相关信息

    ##3.1获取目标hive表文件路径
    for file in hivetabFile:
        if 'dwd_tenant_user' in file:
            goal_hivetabPath = os.path.join(hive_motherDir, file)
            break

    ##3.2读取目标hive表数据
    goal_hivetabData = pd.read_csv(goal_hivetabPath,dtype=str)


    # 3.3 租户相关字典赋值
    rowNum=len(goal_hivetabData['tenant_id'])
    for idx in range(rowNum):
        tenant_id_val = goal_hivetabData['tenant_id'][idx]
        user_id_val = goal_hivetabData['user_id'][idx]
        user_role_val = goal_hivetabData['user_role'][idx]
        user_create_year = str(goal_hivetabData['user_create_time'][idx])[:4]
        if tenant_id_val in tenantDict:
            # tenantDict[tenant_id_val]['studentlist']=[]
            # tenantDict[tenant_id_val]['teacherlist']=[]
            if user_role_val == 'ROLE_STUDENT':
                tenantDict[tenant_id_val]['studentlist'].append({'id': user_id_val, 'year': user_create_year})

            elif user_role_val == 'ROLE_TEACHER':
                tenantDict[tenant_id_val]['teacherlist'].append({'id': user_id_val, 'year': user_create_year})

    ####4.获取表dwd_tenant_course中相关信息

    ##4.1获取目标hive表文件路径
    for file in hivetabFile:
        if 'dwd_tenant_course' in file:
            goal_hivetabPath = os.path.join(hive_motherDir, file)
            break

    ##4.2读取目标hive表数据
    goal_hivetabData = pd.read_csv(goal_hivetabPath,dtype=str)

    # 4.3 租户相关字典赋值
    rowNum = len(goal_hivetabData['tenant_id'])
    for idx in range(rowNum):
        tenant_id_val = goal_hivetabData['tenant_id'][idx]
        course_name_val = goal_hivetabData['course_name'][idx]
        course_auth_status_val = goal_hivetabData['course_auth_status'][idx]    #租户课程授权状态
        course_create_year = goal_hivetabData['data_create_time'][idx][:4] #租户课程实际使用时间，区别于授权开始时间

        if tenant_id_val in tenantDict:



            if course_auth_status_val == 'TRIAL':
                tenantDict[tenant_id_val]['trial_courselist'].append({'id': course_name_val, 'year': course_create_year})
            elif course_auth_status_val == 'NORMAL':
                tenantDict[tenant_id_val]['normal_courselist'].append({'id': course_name_val, 'year': course_create_year})
            else:
                tenantDict[tenant_id_val]['free_courselist'].append({'id': course_name_val, 'year': course_create_year})

    ####5.获取表dwd_course_teach_plan_detail中相关信息

    ##5.1获取目标hive表文件路径
    for file in hivetabFile:
        if 'dwd_course_teach_plan_detail' in file:
            goal_hivetabPath = os.path.join(hive_motherDir, file)
            break

    ##5.2读取目标hive表数据
    goal_hivetabData = pd.read_csv(goal_hivetabPath,dtype=str)

    # 5.3 租户相关字典赋值
    rowNum = len(goal_hivetabData['tenant_id'])
    for idx in range(rowNum):
        tenant_id_val = goal_hivetabData['tenant_id'][idx]
        teachplan_name_val=goal_hivetabData['teachplan_name'][idx]
        if tenant_id_val in tenant_id_list:
            tenantDict[tenant_id_val]['teachplan_name'].append(teachplan_name_val)

    ####6.获取表dwd_course_teach_activity_detail中相关信息

    ##6.1获取目标hive表文件路径
    for file in hivetabFile:
        if 'dwd_course_teach_activity_detail' in file:
            goal_hivetabPath = os.path.join(hive_motherDir, file)
            break

    ##6.2读取目标hive表数据
    goal_hivetabData = pd.read_csv(goal_hivetabPath,dtype=str)

    # 6.3 租户相关字典赋值
    rowNum = len(goal_hivetabData['tenant_id'])
    for idx in range(rowNum):
        tenant_id_val = goal_hivetabData['tenant_id'][idx]
        activity_type_val=goal_hivetabData['activity_type'][idx]
        if tenant_id_val in tenant_id_list:
            tenantDict[tenant_id_val]['activity_type'].append(activity_type_val)

    np.save(os.path.join(hive_motherDir,'hive_tenantDict.npy'),tenantDict)


    endT=time.time()
    print('租户字典构建完成！！！,用时 {}s'.format(endT-startT))


if __name__ == '__main__':
    tenantDict_Construct_F()