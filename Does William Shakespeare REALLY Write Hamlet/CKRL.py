# coding=gb2312
# 王冬虎
# 学习时间:2022/10/2 00:23
#===========基本库import============
import torch
import codecs
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#=========自定义文件import============
from progress_utils import Data_utils
from dataloader import TripleDataset

class CKRL(nn.Module):
    def __init__(self,entity_id_file_path,relation_id_file_path,w2v_dim,margin=1.0, L=2):
        super(CKRL,self).__init__()
        self.model = "CKRL"
        self.w2v_dim = w2v_dim

        #γ ( used to L = ∑ ∑ max( (E(h,r,l)-E(h',r',l')+γ)*C(h,r,l) ), 0) )
        self.margin = margin

        # L2 dis:||h+r-l||
        assert (L == 1 or L == 2)
        self.L = L
        self.distfn = nn.PairwiseDistance(L)

        data_utils = Data_utils(entity_id_file_path,relation_id_file_path)
        self.entityEmbedding,self.relationEmbedding = data_utils.get_embedding_object(self.w2v_dim)

    '''
    get E(h,r,l) and E(h',r',l')-----------------------------------------
    This function used to calculate E score, steps follows:
    ==> Step1: Split input as head, relation and tail index column
    ==> Step2: Transform index tensor to embedding tensor
    ==> Step3: Sum head, relation and tail tensors with weights (1, 1, -1)
    ==> Step4: Calculate distance as final score
    '''
    def scoreOp(self, inputTriple):
        # Step1
        # head : shape(batch_size, 1)
        # relation : shape(batch_size, 1)
        # tail : shape(batch_size, 1)
        head, relation, tail = torch.chunk(input=inputTriple,
                                           chunks=3,
                                           dim=1)
        # Step2
        # head : shape(batch_size, 1, embedDim)
        # relation : shape(batch_size, 1, embedDim)
        # tail : shape(batch_size, 1, embedDim)
        head_embedding = torch.squeeze(self.entityEmbedding(head), dim=1)
        tail_embedding = torch.squeeze(self.entityEmbedding(tail), dim=1)
        relation_embedding = torch.squeeze(self.relationEmbedding(relation), dim=1)

        # Step3 and Step4
        # E : shape(batch_size, embedDim) ==> shape(batch_size, 1)
        E = self.distfn(head_embedding+relation_embedding, tail_embedding)

        return E

    '''
    In every training epoch, the entity embedding should be normalize
    first. There are three steps:
    ==> Step1: Get numpy.array from embedding weight
    ==> Step2: Normalize array
    ==> Step3: Assign normalized array to embedding
    '''
    def normalizeEmbedding(self):
        embedWeight = self.entityEmbedding.weight.detach().cpu().numpy()
        embedWeight = embedWeight / np.sqrt(np.sum(np.square(embedWeight), axis=1, keepdims=True))
        self.entityEmbedding.weight.data.copy_(torch.from_numpy(embedWeight))

    def retEvalWeights(self):
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy()}

    '''
    Input description:
    ==> posX : (torch.tensor)The positive triples tensor, shape(batch_size, 3)
    ==> negX : (torch.tensor)The negative triples tensor, shape(batch_size, 3)
    --------------------------------------------------------------------------
    ==> α:(α ∈ (0, 1))a hyper-parameters that control the ascend or descend pace of local triple confidence,with the assurance that LT (h, r, t) ∈ (0, 1]
    ==> β:(β>0)a hyper-parameters that control the ascend or descend pace of local triple confidence,with the assurance that LT (h, r, t) ∈ (0, 1]
    ==> LT:(LT (h, r, t) ∈ (0, 1]) Local Triple Confidence
    '''
    def forward(self, posX, negX,α,β,LT=1,):
        size = posX.size()[0]
        #get E(h,r,t) score and E(h',r',t') score
        posEScore = self.scoreOp(posX)
        negEScore = self.scoreOp(negX)

        #get C(h,r,t) score-------------------------------
        #Local Triple Confidence:LT
        Q = -(posEScore - negEScore + self.margin)
        judge_LT = lambda Q:[α*LT,LT+β][Q <= 0]
        LT = judge_LT(Q)

        #Global Path Confidence:Prior Path Confidence PP
        p_i = self.get_sample

        #Global Path Confidence:Adaptive Path Confidence AP





        # Get margin ranking loss
        # L= ∑ max(posScore-negScore+margin, 0)*C(h,r,l)
        return torch.sum(F.relu(input=posEScore-negEScore+self.margin)) / size

    '''
    Used to load pretraining entity and relation embedding.
    Implementation steps list as following:
    Method one: (Assign the pre-training vector one by one)
    ==> Step1: Read one line at a time, split the line as entity string and embed vector.
    ==> Step2: Transform the embed vector to np.array
    ==> Step3: Look up entityDict, find the index of the entity from entityDict, assign 
               the embed vector from step1 to the embedding matrix
    ==> Step4: Repeat steps above until all line are checked.
    Method two: (Assign the pre-training at one time)
    ==> Step1: Initial a weight with the same shape of the embedding matrix
    ==> Step2: Read every line of the EmbedFile and assign the vector to the initialized 
               weight.
    ==> Step3: Assign the initialized weight to the embedding matrix at one time after
               all line are checked.
    '''
    def initialWeight(self, entityEmbedFile, entityDict, relationEmbedFile, relationDict):
        print("="*20+"INFO : Loading entity pre-training embedding."+"="*20)
        with codecs.open(entityEmbedFile, "r", encoding="gb2312") as fp:
            _, embDim = fp.readline().strip().split()
            assert int(embDim) == self.entityEmbedding.weight.size()[-1]
            for line in fp:
                ent, embed = line.strip().split("\t")
                embed = np.array(embed.split(","), dtype=float)
                if ent in entityDict:
                    self.entityEmbedding.weight.data[entityDict[ent]].copy_(torch.from_numpy(embed))
        print("="*20+"INFO : Loading relation pre-training embedding."+"="*20)
        with codecs.open(relationEmbedFile, "r", encoding="gb2312") as fp:
            _, embDim = fp.readline().strip().split()
            assert int(embDim) == self.relationEmbedding.weight.size()[-1]
            for line in fp:
                rel, embed = line.strip().split("\t")
                embed = np.array(embed.split(","), dtype=float)
                if rel in entityDict:
                    self.relationEmbedding.weight.data[relationDict[rel]].copy_(torch.from_numpy(embed))

if __name__ == "__main__":
    print("="*20+"test"+"="*20)
