# -*- coding: UTF-8 -*-
# 王冬虎
# 学习时间:2022/10/1 0:23
#===========基本库import============
import torch
import json
import networkx as nx
import matplotlib.pyplot as plt
#from CKRL import CKRL
#from config import TrainConfig
#===========自定义文件import==============
from progress_utils import Data_utils

class Networkx_utils:
    def __init__(self, data_file_path, args):
        self.args =args
        self.graph = self.args.get_nx_graph(data_file_path,show=False)

    def k_shell(self):
        """
        根据Kshell算法计算节点的重要性
        :return: importance_dict{ks:[nodes]}
        """
        importance_dict = {}
        ks = 1
        # print(graph.nodes)
        while self.graph.nodes():
            # 暂存度为ks的顶点
            temp = []
            # 暂存顶点和度对应
            make_degrees_dict = lambda x: {tuple[0]: tuple[1] for tuple in x}
            # print("node_degrees_dict",graph.degree())
            node_degrees_dict = make_degrees_dict(self.graph.degree())
            # 每次删除度值最小的节点而不能删除度为ks的节点否则产生死循环。这也是这个算法存在的问题。
            kks = min(node_degrees_dict.values())

            while True:
                for k, v in node_degrees_dict.items():
                    if v == kks:
                        temp.append(k)
                        self.graph.remove_node(k)
                node_degrees_dict = make_degrees_dict(self.graph.degree())
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
        :return: 所有简单路径组成的list
        """
        all_simple_paths_iter = nx.all_simple_paths(self.graph, source=h, target=t)
        return [simple_path for simple_path in all_simple_paths_iter]

    def PCRA(self, h, t):
        E_l_list = list(self.graph.nodes())
        E_l_list.remove(h)
        nstart = {n:0 for n in E_l_list}
        nstart[h] = 1
        node_pagerank_dict = nx.pagerank(self.graph,nstart=nstart)
        "return R(h,P,t)"
        print(node_pagerank_dict[t])
        return node_pagerank_dict[t]


if __name__ == "__main__":

    print("="*20+"test"+"="*20)
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 4), (2, 4), (3, 4), (4, 5), (5, 6), (5, 7), (3, 5), (6, 7),(1,2),(2,3),(1,3),(3,6),(6,5)])
    #nx.draw(graph,node_size=300,width=0.2,with_labels=True) #pip install decorator==4.4.2
    #plt.show()
    #graph.remove_node(1)
    #print(type(list(graph.nodes())))
    #print(k_shell(graph))
    #node_pagerank_dict = nx.pagerank(graph,nstart=)

    config_json_date = open("path.json", "r", encoding="gb2312")
    json_config = json.loads(config_json_date.read())
    base_path = json_config["base_path"]
    arg = Data_utils(entity_id_file_path=base_path["entity_id_file_path"],relation_id_file_path=base_path["relation_id_file_path"])
    nx_utils = Networkx_utils(data_file_path=base_path["train_data_file_path"],args=arg)
    nx_utils.PCRA(h=13692,t=4132)

    """
    arg = TrainConfig()
    model = CKRL(entity_id_file_path=arg.entity_id_file_path,
         relation_id_file_path=arg.relation_id_file_path,
         w2v_dim=100)
    print(model.entityEmbedding.weight.data[10])
    """