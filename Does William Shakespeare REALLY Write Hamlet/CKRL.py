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
from nx_utils import Networkx_utils
from dataloader import TripleDataset

class CKRL(nn.Module):
    def __init__(self,entity_id_file_path,relation_id_file_path,data_file_path,w2v_dim,margin=1.0,L=2):
        super(CKRL,self).__init__()
        self.model = "CKRL"
        self.w2v_dim = w2v_dim
        # L2 dis:||h+r-l||
        assert (L == 1 or L == 2)
        self.L = L
        self.distfn = nn.PairwiseDistance(L)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #utils object
        self.data_utils = Data_utils(entity_id_file_path,relation_id_file_path)
        self.nx_utils = Networkx_utils(data_file_path=data_file_path, args=self.data_utils)
        #embedding object
        self.entityEmbedding,self.relationEmbedding = self.data_utils.get_embedding_object(self.w2v_dim)

        # γ ( used to L = ∑ ∑ max( (E(h,r,l)-E(h',r',l')+γ)*C(h,r,l) ), 0) )
        self.margin = margin
        # LT Initial value
        self.LT = 1
        self.sigmoid = nn.Sigmoid()

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
    ==> alpha:(α ∈ (0, 1))a hyper-parameters that control the ascend or descend pace of local triple confidence,with the assurance that LT (h, r, t) ∈ (0, 1],default is 0.9
    ==> beta:(β>0)a hyper-parameters that control the ascend or descend pace of local triple confidence,with the assurance that LT (h, r, t) ∈ (0, 1],default is 0.0001
    ==> sigma:a hyper-parameter of Q_pp for smoothing.default is 0.8
    ==> lambda1,lambda2,lambda3:the hyper-parameters for control overall triple confidence,default is 1.5,0.1,0.4
    '''
    def forward(self, posX, negX,alpha,beta,sigma,lambda1,lambda2,lambda3):
        size = posX.size()[0]               #batch size
        #get E(h,r,t) score and E(h',r',t') score
        posEScore = self.scoreOp(posX)
        negEScore = self.scoreOp(negX)

        #get C(h,r,t) score-------------------------------
        #Local Triple Confidence:LT
        Q = -(posEScore - negEScore + self.margin).cpu()

        LT = np.ones(size)  #Initial LT value = [1,1,1,1.....] size=(batchsize,1)
        def judge_LT(q,lt):
            mask_array = torch.gt(q,torch.zeros(size,1)).numpy()
            for mask_id,mask in enumerate(mask_array):
                if mask.any():
                    lt[mask_id] += beta
                else:
                    lt[mask_id] *= alpha
            return torch.from_numpy(lt)
        LT = judge_LT(Q,LT).cuda()  #size = (batch_size,1) ,the vector of this batch's LT

        #judge_LT = np.frompyfunc(lambda Q:[alpha*LT,LT+beta][Q.gt(0)],1,1)  #α=0.9,β=0.0001,此函数类似于pd.apply

        #Global Path Confidence:Prior Path Confidence PP
        head_batch_tensor,relation_batch_tensor,tail_batch_tensor = torch.chunk(input=posX,chunks=3,dim=1)
        head_batch_array,relation_batch_array,tail_batch_array = head_batch_tensor.cpu().numpy(),relation_batch_tensor.cpu().numpy(),tail_batch_tensor.cpu().numpy()
        PP_i_list = []
        AP_i_list = []
        for head,relation,tail in zip(head_batch_array,relation_batch_array,tail_batch_array):     # one sample's head and tail
            PP_i = 0
            AP_i = 0
            p_i_list = self.nx_utils.search_path(int(head),int(tail))
            r_embedding = torch.squeeze(self.relationEmbedding(torch.tensor(relation).cuda()),dim=1)
            for p_i in p_i_list:
                Q_PP = sigma + (1 - sigma) * 1 / len(p_i_list)
                R_pi = self.nx_utils.PCRA(p_i)
                PP_i += Q_PP*R_pi

                r_i_tensor = torch.tensor(self.nx_utils.search_relation(p_i)).cuda()
                r_i_embedding = torch.squeeze(self.relationEmbedding(r_i_tensor), dim=1)
                Q_AP = float(self.distfn(r_embedding,torch.sum(r_i_embedding, dim=0)))
                if Q_AP != 0:
                    AP_i += R_pi/Q_AP
            AP_i = self.sigmoid(torch.tensor(AP_i))
            PP_i_list.append(PP_i)
            AP_i_list.append(AP_i)

        PP = torch.tensor(PP_i_list).cuda()        # a batch's PP tensor

        #Global Path Confidence:Adaptive Path Confidence AP
        AP = torch.tensor(AP_i_list).cuda()        # a batch's AP tensor

        #Triple confidence C(h,r,t)
        C = (lambda1*LT + lambda2*PP + lambda3*AP)  # a batch's C(h,r,t)

        # Get margin ranking loss
        # L= ∑ max(posScore-negScore+margin, 0)*C(h,r,l)
        return torch.sum(F.relu(input=(posEScore-negEScore+self.margin)*C)) / size

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
