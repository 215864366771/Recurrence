������
ѧϰʱ��:2022/10/10 13:54

# **Does William Shakespeare REALLY Write Hamlet?

### _Knowledge Representation Learning with Confidence ����_**

#### **environment:**
* Anaconda
* python            3.8
* torch             1.10.2+cu113
* decorator         4.4.2
* networkx          2.5
* torch-geometric   2.1.0.post1

_note:torch-geometric��Ҫװ4��dependency������dependency�ļ������ṩpython3.8��3.9�汾����,��Anaconda Prompt cd�������ļ�λ�ú�ִ�� pip install Tab����,�ĸ�����װ�ú���ִ��pip install torch-geometric,����torch�汾��ͬ,����Ҫ�����µ�ַ��ѡ����torch�汾���������� ���ص�ַ:[https://pytorch-geometric.com/whl/](https://pytorch-geometric.com/whl/)_

#### **introduce**

##### Dataset:
###### FB15K-237���ݼ�
1. λ��`divided dataset�ļ���`�µ�`train.txt`, `valid.txt`, `test.txt`�乹�ɽṹΪ:ͷʵ��\t��ϵ\tβʵ�塣�ֱ�Ϊѵ����,��֤��,���Լ���
2. `entity2id.txt`,`relation2id.txt`���ɽṹΪ:ʵ��\tid��Ϊ����ʵ��,��ϵ����Ӧ��id
3. λ��`pretrain vector �ļ���`�µ�`FB15K_TransE_Entity2Vec_100.txt`,`FB15K_TransE_Relation2Vec_100.txt`�乹�ɽṹΪ:ʵ��\tfloat����,float����,....(100��),��ʾ�Ѿ���TransEѵ���õ�100άʵ��,��ϵ����
4. `onehot.txt`���ɽṹΪ:int����\tint����\tint����,Ϊprogress_utils.py�ж����Data_utils���one_hot��������,��Ϊ����������ݼ�����onehot����(������չʾ,Train.py��ִ��ѵ��ʱֻ��Ҫ�������ݼ��ļ�·������Զ�ִ��dataloader.py�е�Transform2Index�������ж��ȱ�����ٽ���embedding����)
5. `path.json`Ϊ��������ļ�·��,���������ļ����ƻ�·�������޸ĺ�ֻ���`path.json`�ļ��ж�Ӧkeyֵ�µ�value�����޸ļ���
##### utils:
###### progress_utils.py
�ṩ��Data_utils���Evaluation��,Data_utils����Ҫ�����ݽ��д�����
1. one_hot�����������Dataset��1.�����ļ�����Dataset��2.�ļ��й涨��id����onehot����,����save=Tureʱ�����ļ�Dataset��3.����onehot.txt
2. get_embedding_object���������nn.Embedding��������embedding����
3. get_geometric_graph������������Dataset��1.�����ļ�����torch_geometric�����ṩ��Dataset��,��ͬ������torch��д,������Dataloader��
4. get_nx_graph������������Dataset��1.�����ļ������networkx���ṩ��graph��ʵ������.

Evaluation��Ϊ��������,��������ѵ��������������
1. MREvaluation_CKRL�������model����embedding������ó���������������,����MeanRank����,��һ��epoch��ƽ��rank��rank�����ں�������������д��,����Ϊ:��embed����h��r���������l�����ƶ�(���û��ھ���),��||h+r-t||

###### nx_utils.py
�ṩ��������ݽ���networkx���ṩ��graph��ʵ������
1. [ ] (!!δ���!!)�������networkx.graph��ʵ����������ִ��kshell�㷨
2. [x] �������networkx.graph��ʵ�������ͷʵ��βʵ�����뷵��������·��,��������PCRA�㷨�е���Դ��
##### dataloader:
###### dataloader.py
**�ṩ�˽�����תΪ����torch.Dataset���TripleDataset��,��������Dataset��1.����ж��ȱ���,�����ճ��������ʽ���ͷʵ��,βʵ��,��ϵ�����滻,���__getitem__()���������Ͷ�Ӧ�ĸ���
1. generateNegSamples����������ָ���������ɸ���,������ɵĸ���DataFrame���ݱ�����self.negDf����������
2. ��̬transformToIndex�������������Dataset��1.����ж��ȱ���,��TripleDataset��ʵ������ʱ����ô˷������������ݵ����������DataFrame��ʽ
3. __getitem__()������Ϊtorch.Dataset������������,������dataloader�������batch������᷵��һ������һ������(CKRL��modelֱ�ӷ��������е�L,������Ҫ��E(h,r,t)-E(h',r',t'),ֱ��һ�����Ͷ�Ӧ������������CKRL.py�е�scoreOp�������)
**##### model:**
###### CKRL.py
CKRL.py�ṩ��CKRL��,Ϊ������CKRL���,��Ϊ�̳���nn.Module����,���������Զ�dataloader.py�й����nn.Dataset��תΪnn.Dataloader�������model�����Ϊ������L��ʽ(����ʧ����,δ���)
1. [x] scoreOp���������������Ԫ������������������E
2. [x] normalizeEmbedding������Ϊ�����embed����������й�һ������,��ֹ���򴫲������ݶ�����
3. [x] initialWeight����������ÿ��ѵ��ǰ�ɽ�ԭ��embed�ú�����������ļ�����nn.embedding����,�̳�ԭ��ѵ��
4. [ ] (!!!δ���!!!) forward������ģ�����������е���ʧ����,������Train.py��ֱ�Ӷ�ģ���������optimizer�Ż�����

_note:initialWeight������ǰѵ��embed��ʽΪTrain.py�е�savemodel������д��ʽ,Ϊ

entity_num entity_dim

entity_name\t��������,��������,��������,��������,......  (��������Ϊfloat������)
    .
    .
    .
���Ҫ����Dataset�е�3.�е�TransEԤѵ���õĲ���,����Ҫע�͵�CKRL.py��130�е�132��,139�е�141�л�����`FB15K_TransE_Entity2Vec_100.txt`��`FB15K_TransE_Relation2Vec_100.txt`��һ�м���entity/relation����\t100_
###### config.py
�ṩ��TrainConfig��,������������Դ����Train.py������Ҫ�Ĳ���
###### Train.py
�ṩ��ѵ��������֤����dataloader�����������Լ�һ��TrainTriples��,�������ṩ�����·���
1. [x] creat_data����Ϊ���ô�ҳ�е�ѵ��������֤��dataloader������������������ѵ���ú���֤�õ�Dataloader��ʵ������
2. [ ] (!!δ��ɣ���)creat_model����Ϊ����CKRL.py,������һ���̳���nn.Model��CKRL��ʵ������,������ΪDataLoader��ʵ��,���ΪL
3. [x] load_pretrain_embedding��������CKRL.py��initialWeight����,��fit����ִ��ǰִ�д˷������Խ�ָ��·����ѵ���õ���������
4. [x] load_pretrain_model��������ǰfit��ִ��savemodel������ŵ��ļ�·�����ص��˴�ѵ����
5. [ ] (!!δ��ɣ���)



