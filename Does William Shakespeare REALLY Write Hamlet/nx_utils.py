# -*- coding: UTF-8 -*-
# 王冬虎
# 学习时间:2022/10/1 0:23
#===========基本库import============
import torch
import json
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
#from CKRL import CKRL
#from config import TrainConfig
#===========自定义文件import==============
from progress_utils import Data_utils

class Networkx_utils:
    def __init__(self, data_file_path, args):
        self.args =args
        self.graph = self.args.get_nx_graph(data_file_path,show=False)

        self.make_degrees_dict = lambda x: {tuple[0]: tuple[1] for tuple in x}


    def k_shell(self):
        """
        根据Kshell算法计算节点的重要性
        :return: importance_dict{ks:[nodes]}
        """
        importance_dict = {}
        ks = 1
        copy_graph = deepcopy(self.graph)
        # print(graph.nodes)
        while self.graph.nodes():
            # 暂存度为ks的顶点
            temp = []
            # 暂存顶点和度对应
            node_degrees_dict = self.make_degrees_dict(copy_graph.degree())
            # 每次删除度值最小的节点而不能删除度为ks的节点否则产生死循环。这也是这个算法存在的问题。
            kks = min(node_degrees_dict.values())

            while True:
                for k, v in node_degrees_dict.items():
                    if v == kks:
                        temp.append(k)
                        copy_graph.remove_node(k)
                node_degrees_dict = self.make_degrees_dict(copy_graph.degree())
                if node_degrees_dict is None or kks not in node_degrees_dict.values():
                    break
            importance_dict[ks] = temp
            ks += 1
        return importance_dict

    def found_path(self,h,t):
        """
        :param graph: nx的graph对象
        :param h:头实体编码
        :param t: 尾实体编码
        :return: 所有简单路径组成的list,简单路径数量
        """
        all_simple_paths_iter = nx.all_simple_paths(self.graph, source=h, target=t)
        return list(all_simple_paths_iter)


    def PCRA(self,p_i):
        """
        E_l_list = list(self.graph.nodes())
        E_l_list.remove(h)
        nstart = {n:0 for n in E_l_list}
        nstart[h] = 1
        node_pagerank_dict = nx.pagerank(self.graph,nstart=nstart)
        "return R(h,P,t)"
        print(node_pagerank_dict[t])
        return node_pagerank_dict[t]
        """
        resource = 1
        node_out_degree_dict = self.make_degrees_dict(self.graph.out_degree())
        for node_label in p_i[:-1]:
            node_out_degree = node_out_degree_dict[node_label]
            resource *= 1/node_out_degree
        return resource  #从此路径到达尾实体的资源量


if __name__ == "__main__":

    print("="*20+"test"+"="*20)
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(ebunch_to_add=[(1,4,1), (2,4,1), (3,4,1), (4,5,1), (5,6,1), (5,7,1), (3,5,1), (6,7,1),(1,2,1),(2,3,1),(1,3,1),(3,6,1),(6,5,1)])
    #print(list(nx.all_simple_paths(graph, source=1, target=4)))
    #nx.draw(graph,node_size=300,width=0.2,with_labels=True) #pip install decorator==4.4.2
    #plt.show()

    #node_pagerank_dict = nx.pagerank(graph,nstart=)

    config_json_date = open("path.json", "r", encoding="gb2312")
    json_config = json.loads(config_json_date.read())
    base_path = json_config["base_path"]
    arg = Data_utils(entity_id_file_path=base_path["entity_id_file_path"],relation_id_file_path=base_path["relation_id_file_path"])
    nx_utils = Networkx_utils(data_file_path=base_path["train_data_file_path"],args=arg)
    #print(nx_utils.node_out_degree_dict)
    #print(type(nx_utils.found_path(h=13692,t=4132)))
    #print(nx.all_simple_paths(graph, source=13692, target=4132))
    for path in nx_utils.found_path(h=13692,t=4132):
        print(nx_utils.PCRA(path))
    #path = nx_utils.found_path(h=13692,t=4132)
    #print(path)
    #for p_i in path:
        #print(nx_utils.PCRA(p_i))

    """
    arg = TrainConfig()
    model = CKRL(entity_id_file_path=arg.entity_id_file_path,
         relation_id_file_path=arg.relation_id_file_path,
         w2v_dim=100)
    print(model.entityEmbedding.weight.data[10])
    """