# coding=gb2312
# 王冬虎
# 学习时间:2022/10/8 10:17
#===========基本库import============
import os
import json
import numpy as np
import pandas as pd
import codecs
import pickle
import torch
import torch.nn as nn
from torchvision import transforms,models,datasets
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from tensorboardX import SummaryWriter


#==========自定义文件import==========
from progress_utils import Data_utils,Evaluation
from dataloader import TripleDataset
from CKRL import CKRL
from config import *

def prepare_dataloader(args, repSeed, exSeed, headSeed, tailSeed):
    dataset = TripleDataset(entity_id_file_path=args.entity_id_file_path,
                            relation_id_file_path=args.relation_id_file_path,
                            data_file_path=args.train_data_file_path)
    dataset.generateNegSamples(repProba=args.repproba,
                               exProba=args.exproba,
                               repSeed=repSeed,
                               exSeed=exSeed,
                               headSeed=headSeed,
                               tailSeed=tailSeed)
    dataloader = DataLoader(dataset,
                            batch_size=args.batchsize,
                            shuffle=args.shuffle,
                            num_workers=args.numworkers,
                            drop_last=args.droplast)
    return dataloader

def prepare_eval_dataloader(args):
    dataset = TripleDataset(data_file_path=args.valid_data_file_path,
                            entity_id_file_path=args.entity_id_file_path,
                            relation_id_file_path=args.relation_id_file_path)
    dataloader = DataLoader(dataset,
                            batch_size=len(dataset),
                            shuffle=False,
                            drop_last=False)
    return dataloader

def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

class TrainTriples:
    def __init__(self, args):
        self.args = args
        self.data_utils = Data_utils(entity_id_file_path = args.entity_id_file_path,
                                    relation_id_file_path = args.relation_id_file_path)

    def creat_data(self):
        print("="*20+"INFO : Prepare dataloader"+"="*20)
        self.dataloader = prepare_dataloader(self.args,0,0,0,0)
        self.evalloader = prepare_eval_dataloader(self.args)
        self.entityDict = self.data_utils.entity_id_dict
        self.relationDict = self.data_utils.relation_id_dict

    def creat_model(self):
        print("="*20+"INFO : Init model %s"%self.args.modelname+"="*20)
        if self.args.modelname == "CKRL":
            self.model = CKRL(entity_id_file_path = self.args.entity_id_file_path,
                              relation_id_file_path = self.args.relation_id_file_path,
                              data_file_path =self.args.train_data_file_path,
                              w2v_dim = self.args.CKRL["EmbeddingDim"],
                              margin = self.args.CKRL["Margin"],
                              L = self.args.CKRL["L"]
                              )
        else:
            print("ERROR : No model named %s"%self.args.modelname)
            exit(1)
        if self.args.usegpu:
            with torch.cuda.device(0):
                self.model.cuda()

    def load_pretrain_embedding(self):
        if self.args.modelname == "CKRL":
            print("="*20+"INFO : Loading pre-training entity and relation embedding"+"="*20)
            self.model.initialWeight(entityEmbedFile=self.args.entity_vec_file_path,
                                     entityDict=self.data_utils.entity_id_dict,
                                     relationEmbedFile=self.args.relation_vec_file_path,
                                     relationDict=self.data_utils.relation_id_dict)
        else:
            print("ERROR : Model %s is not supported!"%self.args.modelname)
            exit(1)

    def load_pretrain_model(self):
        if self.args.modelname == "CKRL" and os.path.exists(self.args.premodel):
            print("="*20+"INFO : Loading pre-training model."+"="*20)
            modelType = os.path.splitext(self.args.premodel)[-1]
            if modelType == ".param":
                self.model.load_state_dict(torch.load(self.args.premodel))
            elif modelType == ".model":
                self.model = torch.load(self.args.premodel)
            else:
                print("ERROR : Model type %s is not supported!")
                exit(1)
        else:
            print("ERROR : Model %s is not exist!" % self.args.modelname)
            exit(1)

    def fit(self):
        """
        epoch:
        learning_rate:
        optimizer:Adam
        evaluator:meanRank

        :return:
        """
        EPOCHS = self.args.epochs
        LR = self.args.learningrate
        OPTIMIZER = self.args.optimizer
        if OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         weight_decay=self.args.weightdecay,
                                         lr=LR)
        else:
            print("ERROR : Optimizer %s is not supported."%OPTIMIZER)
            exit(1)

        # Training, GLOBAL-STEP and GLOBAL-EPOCH are used for summary
        minLoss = float("inf")
        bestMR = float("inf")
        GLOBALSTEP = 0
        GLOBALEPOCH = 0
        for seed in range(100):
            print("INFO : Using seed %d" % seed)
            self.dataloader = prepare_dataloader(self.args, repSeed=seed, exSeed=seed, headSeed=seed, tailSeed=seed)
            self.evalloader = prepare_eval_dataloader(self.args)
            for epoch in range(EPOCHS):
                GLOBALEPOCH += 1
                STEP = 0
                print("="*20+"EPOCHS(%d/%d)"%(epoch+1, EPOCHS)+"="*20)
                for posX, negX in self.dataloader:
                    # Allocate tensor to devices
                    if self.args.usegpu:
                        with torch.cuda.device(0):
                            posX = Variable(torch.LongTensor(posX).cuda())
                            negX = Variable(torch.LongTensor(negX).cuda())
                    else:
                        posX = Variable(torch.LongTensor(posX))
                        negX = Variable(torch.LongTensor(negX))
                    # Normalize the embedding if necessary
                    self.model.normalizeEmbedding()

                    # Calculate the loss from the model
                    """
                    一般loss = nn.BCELoss()
                    loss(score,target)这个loss也是由nn.module写的,forward所return回的值则是loss,所以这里直接model返回CKRL的Loss(en)进行优化即可
                    """
                    loss = self.model(posX,negX,alpha=0.9,beta=0.0001,sigma=0.8,lambda1=1.5,lambda2=0.1,lambda3=0.4)
                    if self.args.usegpu:
                        lossVal = loss.cuda().item()
                    else:
                        lossVal = loss.item()

                    # Calculate the gradient and step down
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Print information and add to summary
                    if minLoss > lossVal:
                        minLoss = lossVal
                    print("[TRAIN-EPOCH(%d/%d)-STEP(%d)]Loss:%.4f, minLoss:%.4f"%(epoch+1, EPOCHS, STEP, lossVal, minLoss))
                    STEP += 1
                    GLOBALSTEP += 1
                    sumWriter.add_scalar('train/loss', lossVal, global_step=GLOBALSTEP)
                if GLOBALEPOCH % self.args.lrdecayepoch == 0:
                    adjust_learning_rate(optimizer, decay=self.args.lrdecay)
                if GLOBALEPOCH % self.args.evalepoch == 0:
                    MR = Evaluation().MREvaluation_CKRL(evalloader=self.evalloader,
                                                        simMeasure=self.args.simmeasure,
                                                        **self.model.retEvalWeights()) #self.model.retEvalWeights():get model.weight by dict
                    sumWriter.add_scalar('train/eval', MR, global_step=GLOBALEPOCH)
                    print("[EVALUATION-EPOCH(%d/%d)]Measure method %s, eval %.4f"% \
                          (epoch+1, EPOCHS, self.args.evalmethod, MR))
                    # Save the model if new MR is better
                    if MR < bestMR:
                        bestMR = MR
                        self.saveModel()
                        self.dumpEmbedding()

    def saveModel(self):
        """
        config里if modelsave == "param"只存model.state_dict()，if modelsave == "full"就存所有
        """
        if self.args.modelsave == "param":
            "only save param"
            path = os.path.join(self.args.modelpath, "{}_ent{}_rel{}.param".format(self.args.modelname, getattr(self.args, self.args.modelname)["EmbeddingDim"], getattr(self.args, self.args.modelname)["EmbeddingDim"]))
            torch.save(self.model.state_dict(), path)
        elif self.args.modelsave == "full":
            "save model + param"
            path = os.path.join(self.args.modelpath, "{}_ent{}_rel{}.model".format(self.args.modelname, getattr(self.args, self.args.modelname)["EmbeddingDim"], getattr(self.args, self.args.modelname)["EmbeddingDim"]))
            torch.save(self.model, path)
        else:
            print("ERROR : Saving mode %s is not supported!"%self.args.modelsave)
            exit(1)

    def dumpEmbedding(self):
        entWeight = self.model.entityEmbedding.weight.detach().cpu().numpy()
        relWeight = self.model.relationEmbedding.weight.detach().cpu().numpy()
        entityNum, entityDim = entWeight.shape
        relationNum, relationDim = relWeight.shape
        entsave = os.path.join(self.args.embedpath, "entityEmbedding.txt")
        relsave = os.path.join(self.args.embedpath, "relationEmbedding.txt")
        """
        dumpEmbedding format:
        entity_num entity_dim
        entity_name\t××××,××××,××××,××××,......  (××××为float的数字)
        
        relation同理
        """
        with codecs.open(entsave, "w", encoding="gb2312") as fp:
            fp.write("{} {}\n".format(entityNum, entityDim))
            for ent, embed in zip(self.entityDict.keys(), entWeight):
                fp.write("{}\t{}\n".format(ent, ",".join(embed.astype(np.str))))
        with codecs.open(relsave, "w", encoding="gb2312") as fp:
            fp.write("{} {}\n".format(relationNum, relationDim))
            for rel, embed in zip(self.relationDict.keys(), relWeight):
                fp.write("{}\t{}\n".format(rel, ",".join(embed.astype(np.str))))

if __name__ == "__main__":
    #Prepare args
    args = TrainConfig()
    #Print args
    args.print_args()
    sumWriter = SummaryWriter(log_dir=args.summarydir)
    #Prepare data and model
    model = TrainTriples(args)
    model.creat_data()
    model.creat_model()
    #confirm whether to use pre-train  embedding
    if args.loadembed:
        model.load_pretrain_embedding()
    model.fit()

    sumWriter.close()
