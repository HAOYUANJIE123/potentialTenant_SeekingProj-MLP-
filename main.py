from FeatureConstructModel.tenant_hiveDataMerge import tenantDict_Construct_F
from FeatureConstructModel.tenantFeatureConstruct import featureDictConstruct_F
from DataAnalysisModel.commonCharacterAnalysis import get_X_data

from labelConstructModel.labelConstruction import virtualLabel_construct_F
from DeepLearningModel.DatasetProcess.dataset_split import datasetSplit_F
from DeepLearningModel.train_main_github import wholeProject_F

#特征各维度直方图绘制
def drawHist_Eachdim_main():
    tenantDict_Construct_F()
    featureDictConstruct_F()
    get_X_data()

def potentialTenant_seeking_F():

    virtualLabel_construct_F(postive_n=20, negative_n=453)  # 根据与正式租户的集合相似度，确定虚拟标签
    datasetSplit_F()
    # wholeProject_F(just_predict=False, predict_set_names=['train', 'test'])#模型训练
    wholeProject_F(just_predict=True, predict_set_names=['train', 'test'])#有标签预测
    wholeProject_F(just_predict=True,predict_set_names=['train', 'test','other'])#无标签预测






if __name__ == '__main__':
    # drawHist_Eachdim_main() #特征各维度直方图绘制
    potentialTenant_seeking_F() #潜在租户发掘工程