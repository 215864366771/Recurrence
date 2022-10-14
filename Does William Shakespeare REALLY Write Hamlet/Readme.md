王冬虎
学习时间:2022/10/10 13:54

# **Does William Shakespeare REALLY Write Hamlet?

### _Knowledge Representation Learning with Confidence 复现_**

#### **environment:**
* Anaconda
* python            3.8
* torch             1.10.2+cu113
* decorator         4.4.2
* networkx          2.5
* torch-geometric   2.1.0.post1

_note:torch-geometric需要装4个dependency，附带dependency文件夹中提供python3.8和3.9版本依赖,打开Anaconda Prompt cd到依赖文件位置后执行 pip install Tab即可,四个依赖装好后再执行pip install torch-geometric,如若torch版本不同,则需要从以下地址挑选符合torch版本的依赖下载 下载地址:[https://pytorch-geometric.com/whl/](https://pytorch-geometric.com/whl/)_

#### **introduce**

##### Dataset:
###### FB15K-237数据集
1. 位于`divided dataset文件夹`下的`train.txt`, `valid.txt`, `test.txt`其构成结构为:头实体\t关系\t尾实体。分别为训练集,验证集,测试集。
2. `entity2id.txt`,`relation2id.txt`构成结构为:实体\tid。为所有实体,关系所对应的id
3. 位于`pretrain vector 文件夹`下的`FB15K_TransE_Entity2Vec_100.txt`,`FB15K_TransE_Relation2Vec_100.txt`其构成结构为:实体\tfloat数字,float数字,....(100个),表示已经用TransE训练好的100维实体,关系向量
4. `onehot.txt`构成结构为:int数字\tint数字\tint数字,为progress_utils.py中定义的Data_utils类的one_hot方法生成,其为对输入的数据集进行onehot编码(仅用作展示,Train.py在执行训练时只需要输入数据集文件路径后会自动执行dataloader.py中的Transform2Index方法进行独热编码后再进行embedding操作)
5. `path.json`为存放上述文件路径,当对上述文件名称或路径进行修改后只需对`path.json`文件中对应key值下的value进行修改即可
##### utils:
###### progress_utils.py
提供了Data_utils类和Evaluation类,Data_utils类主要对数据进行处理方法
1. one_hot方法对输入的Dataset中1.类型文件参照Dataset中2.文件中规定的id进行onehot编码,并且save=Ture时产生文件Dataset中3.类型onehot.txt
2. get_embedding_object方法会调用nn.Embedding方法产生embedding对象
3. get_geometric_graph方法会对输入的Dataset中1.类型文件构造torch_geometric库中提供的Dataset类,其同样基于torch编写,可做成Dataloader类
4. get_nx_graph方法会对输入的Dataset中1.类型文件构造成networkx库提供的graph类实例对象.

Evaluation类为评估函数,用于评估训练好向量的质量
1. MREvaluation_CKRL方法会对model进行embedding操作后得出的向量进行评估,采用MeanRank方法,求一个epoch的平均rank，rank计算在后面两个方法中写出,具体为:求embed出的h和r向量相加与l的相似度(采用基于距离),即||h+r-t||

###### nx_utils.py
提供了针对数据进行networkx库提供的graph类实例操作
1. [ ] (!!未完成!!)将构造的networkx.graph类实例对象输入执行kshell算法
2. [x] 将构造的networkx.graph类实例对象和头实体尾实体输入返回其所有路径,可用于求PCRA算法中的资源量
##### dataloader:
###### dataloader.py
**提供了将数据转为属于torch.Dataset类的TripleDataset类,其对输入的Dataset中1.类进行独热编码,并按照超参数概率进行头实体,尾实体,关系进行替换,最后__getitem__()返回正例和对应的负例
1. generateNegSamples方法即按照指定概率生成负例,最后将生成的负例DataFrame数据保存在self.negDf的类属性中
2. 静态transformToIndex方法即对输入的Dataset中1.类进行独热编码,在TripleDataset类实例构造时便调用此方法将所有数据当做正例变成DataFrame格式
3. __getitem__()方法即为torch.Dataset类最后输出方法,构造完dataloader类后打包成batch后遍历会返回一个正例一个负例(CKRL中model直接返回论文中的L,所以需要求E(h,r,t)-E(h',r',t'),直接一正例和对应负例输入后调用CKRL.py中的scoreOp方法求得)
**##### model:**
###### CKRL.py
CKRL.py提供了CKRL类,为论文中CKRL框架,此为继承自nn.Module容器,此容器可以对dataloader.py中构造的nn.Dataset类转为nn.Dataloader类输入此model后输出为论文中L公式(即损失函数,未完成)
1. [x] scoreOp方法即对输入的三元组算论文中能量函数E
2. [x] normalizeEmbedding方法用为对最后embed后的向量进行归一化操作,防止反向传播出现梯度问题
3. [x] initialWeight方法用于在每次训练前可将原先embed好后的向量保存文件赋给nn.embedding对象,继承原来训练
4. [ ] (!!!未完成!!!) forward方法让模型输入论文中的损失函数,后续在Train.py中直接对模型输入进行optimizer优化即可

_note:initialWeight加载先前训练embed格式为Train.py中的savemodel方法所写格式,为

entity_num entity_dim

entity_name\t××××,××××,××××,××××,......  (××××为float的数字)
    .
    .
    .
如果要加载Dataset中的3.中的TransE预训练好的参数,则需要注释掉CKRL.py中130行到132行,139行到141行或者在`FB15K_TransE_Entity2Vec_100.txt`和`FB15K_TransE_Relation2Vec_100.txt`第一行加上entity/relation个数\t100_
###### config.py
提供了TrainConfig类,此类里的类属性存放了Train.py中所需要的参数
###### Train.py
提供了训练集和验证集的dataloader制作方法，以及一个TrainTriples类,此类中提供了如下方法
1. [x] creat_data方法为调用此页中的训练集和验证集dataloader制作方法制作了两个训练用和验证用的Dataloader类实例对象
2. [ ] (!!未完成！！)creat_model方法为调用CKRL.py,创造了一个继承自nn.Model的CKRL类实例对象,其输入为DataLoader类实例,输出为L
3. [x] load_pretrain_embedding方法调用CKRL.py的initialWeight方法,在fit方法执行前执行此方法可以将指定路径下训练好的向量加载
4. [x] load_pretrain_model方法将先前fit中执行savemodel方法存放的文件路径加载到此次训练中
5. [ ] (!!未完成！！)



