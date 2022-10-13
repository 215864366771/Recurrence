# -*- coding: UTF-8 -*-
# 王冬虎
# 学习时间:2022/10/1 0:23
#===========基本库import============
import torch
import networkx as nx
import matplotlib.pyplot as plt
from CKRL import CKRL
from config import TrainConfig
def kshell(graph):
    """
    根据Kshell算法计算节点的重要性
    :param graph: 图
    :return: importance_dict{ks:[nodes]}
    """
    importance_dict = {}
    ks = 1
    print(graph.nodes)
    while graph.nodes():
        # 暂存度为ks的顶点
        temp = []
        node_degrees_dict = graph.degree()
        #print("node_degrees_dict",graph.degree())
        # 每次删除度值最小的节点而不能删除度为ks的节点否则产生死循环。这也是这个算法存在的问题。
        kks = min(node_degrees_dict.values())
        while True:
            for k, v in node_degrees_dict.items():
                if v == kks:
                    temp.append(k)
                    graph.remove_node(k)
            node_degrees_dict = graph.degree()
            if kks not in node_degrees_dict.values():
                break
        importance_dict[ks] = temp
        ks += 1
    return importance_dict

def found_path(graph,h,l):
    """
    :param graph: nx的graph对象
    :param h:头实体编码
    :param l: 尾实体编码
    :return: 所有简单路径组成的list
    """
    all_simple_paths_iter = nx.all_simple_paths(graph,source=h,target=l)
    return [simple_path for simple_path in all_simple_paths_iter]

    #print(nx.dijkstra_path_length(graph,h,l))

if __name__ == "__main__":
    print("="*20+"test"+"="*20)
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 4), (2, 4), (3, 4), (4, 5), (5, 6), (5, 7), (3, 5), (6, 7),(1,2),(2,3),(1,3),(3,6),(6,5)])
    nx.draw(graph,node_size=300,width=0.2,with_labels=True) #pip install decorator==4.4.2
    plt.show()

    print(found_path(graph,1,5))
    """
    arg = TrainConfig()
    model = CKRL(entity_id_file_path=arg.entity_id_file_path,
         relation_id_file_path=arg.relation_id_file_path,
         w2v_dim=100)
    print(model.entityEmbedding.weight.data[10])
    """