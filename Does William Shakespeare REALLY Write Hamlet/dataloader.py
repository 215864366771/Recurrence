# coding=gb2312
# 王冬虎
# 学习时间:2022/10/1 10:23
#===========基本库import============
import os
import json
import numpy as np
import pandas as pd
import torch
from torchvision import transforms,models,datasets
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
#===========自定义文件import==============
from progress_utils import Data_utils

class TripleDataset(Dataset):
    def __init__(self,entity_id_file_path,relation_id_file_path,data_file_path):
        super(Dataset, self).__init__()
        # get entity-id dict and relation-id dict
        print(20*"="+"INFO : Load entity and relation dict."+"="*20)
        data_utiles = Data_utils(entity_id_file_path=entity_id_file_path,
                                 relation_id_file_path=relation_id_file_path)
        self.entity_id_dict = data_utiles.entity_id_dict
        self.relation_id_dict = data_utiles.relation_id_dict
        # Creat a nx_graph (To be used...)
        self.graph = data_utiles.get_nx_graph(data_file_path=data_file_path,
                                              show=False)
        # Transform entity and relation to id，generate one hot encoding
        print(20*"="+"INFO : Loading positive triples and transform to id."+"="*20)
        self.data_df = pd.read_csv(data_file_path,
                                   sep="\t",
                                   names=["head", "relation", "tail"],
                                   header=None,
                                   encoding="gb2312",
                                   keep_default_na=False)

        self.transformToIndex(self.data_df, repDict={"head":self.entity_id_dict,
                                                     "relation":self.relation_id_dict,
                                                     "tail":self.entity_id_dict})


    def generateNegSamples(self, repProba=0.5, exProba=0.5, repSeed=0, exSeed=0, headSeed=0, tailSeed=0):
        """
        按参数规则生成负例集T(h',r',l')赋给类属性negDf
        """
        assert repProba >= 0 and repProba <= 1.0 and exProba >= 0 and exProba <= 1.0
        # Generate(creat) negative samples from positive samples
        print("="*20+"INFO : Generate negative samples from positive samples."+"="*20)
        self.negDf = self.data_df.copy()
        np.random.seed(repSeed)
        repProbaDistribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.negDf), ))
        np.random.seed(exSeed)
        exProbaDistribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.negDf), ))
        shuffleHead = self.negDf["head"].sample(frac=1.0, random_state=headSeed)
        shuffleTail = self.negDf["tail"].sample(frac=1.0, random_state=tailSeed)

        # Replacing head or tail
        def replaceHead(relHead, shuffHead, shuffTail, repP, exP):
            if repP >= repProba:
                return relHead
            else:
                if exP > exProba:
                    return shuffHead
                else:
                    return shuffTail
        def replaceTail(relTail, shuffHead, shuffTail, repP, exP):
            if repP < repProba:
                return relTail
            else:
                if exP > exProba:
                    return shuffTail
                else:
                    return shuffHead

        #Creat T(h',r',l')
        self.negDf["head"] = list(map(replaceHead, self.negDf["head"], shuffleHead, shuffleTail, repProbaDistribution, exProbaDistribution))
        self.negDf["tail"] = list(map(replaceTail, self.negDf["tail"], shuffleHead, shuffleTail, repProbaDistribution, exProbaDistribution))

    @staticmethod
    def transformToIndex(csvData:pd.DataFrame, repDict:dict):
        """
        相当于one_hot_encoding(按照输入的data_file为顺序)
        """
        for col in repDict.keys():
            csvData[col] = csvData[col].apply(lambda x:repDict[col][x])

    # 2 Class DataSet general operation
    def __len__(self):
        return len(self.data_df)
    def __getitem__(self, item):
        """
        :return:返回 np.array(self.data_df.iloc[item,:3])即为PosX,
                    np.array(self.negDf.iloc[item,:3])即为NegX
        """
        if hasattr(self, "negDf"):
            return np.array(self.data_df.iloc[item,:3]), np.array(self.negDf.iloc[item,:3])
        else:
            return np.array(self.data_df.iloc[item,:3])