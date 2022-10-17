# coding=gb2312
# 王冬虎
# 学习时间:2022/9/30 10:23
#===========基本库import============
import os
import torch
import torch.nn as nn
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import dataloader


class Data_utils():
	def __init__(self, entity_id_file_path, relation_id_file_path):
		"""
		:param entity_id_file_path:数据的实体id文件路径(str,path)
		:param relation_id_file_path:数据的关系id文件路径(str,path)

		制作一个Data_utils对象,此对象专门用于处理数据集
		"""
		self.entity_id_file_path = entity_id_file_path
		self.relation_id_file_path = relation_id_file_path

		self.data_list = []
		self.entity_id_dict = {}
		self.relation_id_dict = {}
		self.one_hot_data_list = []

		entity_id_list = open(self.entity_id_file_path).readlines()
		entitys = [entity_id.split("\t")[0] for entity_id in entity_id_list]
		entity_ids = [entity_id.split("\t")[1][:-1] for entity_id in entity_id_list]
		for entity, entity_id in zip(entitys, entity_ids):
			self.entity_id_dict[entity] = entity_id

		relation_id_list = open(self.relation_id_file_path).readlines()
		relations = [relation_id.split("\t")[0] for relation_id in relation_id_list]
		relation_ids = [relation_id.split("\t")[1][:-1] for relation_id in relation_id_list]
		for relation, relation_id in zip(relations, relation_ids):
			self.relation_id_dict[relation] = relation_id

	def one_hot(self, data_file_path, save, one_hot_file_path):
		"""
		:param data_file_path:要处理的数据文件路径(str,path)

		:param save:是否储存编码后的数据(bool)
		:param one_hot_file_path:one_hot编码后存储到文件路径(str,path)
		"""
		self.data_list = open(data_file_path).readlines()
		h_list = [data.split("\t")[0] for data in self.data_list]
		r_list = [data.split("\t")[1] for data in self.data_list]
		l_list = [data.split("\t")[2][:-1] for data in self.data_list]
		for h, r, l in zip(h_list, r_list, l_list):
			data = ""
			for entity in self.entity_id_dict.keys():
				if h == entity:
					data += self.entity_id_dict[entity] + '\t'
			for relation in self.relation_id_dict.keys():
				if r == relation:
					data += self.relation_id_dict[relation] + '\t'
			for entity in self.entity_id_dict.keys():
				if l == entity:
					data += self.entity_id_dict[entity]
			self.one_hot_data_list.append(data)

		print("=" * 20 + "data after one hot encoding" + "=" * 20 + "\n", self.one_hot_data_list)
		if save:
			if not os.path.exists(one_hot_file_path):
				os.mkdir(one_hot_file_path)
			one_hot_file = open(one_hot_file_path, "w")
			for one_hot_data in self.one_hot_data_list:
				one_hot_file.write(one_hot_data + "\n")
			one_hot_file.close()
			print("=" * 20 + "one hot encoding finished" + "=" * 20)
		return self.data_list, self.one_hot_data_list, self.entity_id_dict, self.relation_id_dict

	def get_embedding_object(self,w2v_dim):
		"""
		:param w2v_dim:embedding维度
		:return: embedding object(entity),embedding object(relation)
		"""
		entity_num_embedding = len(open(self.entity_id_file_path).readlines())
		relation_num_embedding = len(open(self.relation_id_file_path).readlines())
		print("=" * 20, "the number of entity is", entity_num_embedding, "=" * 20)
		print("=" * 20, "the number of relation is", relation_num_embedding, "=" * 20)
		embedding_entity = nn.Embedding(entity_num_embedding, w2v_dim)
		embedding_relation = nn.Embedding(relation_num_embedding, w2v_dim)
		return embedding_entity, embedding_relation

	def get_geometric_graph(self, entity_vec_file_path, relation_vec_file_path,data_file_path):
		"""
		:param entity_vec_file_path: 预训练好的实体向量文件路径(str,path)
		:param relation_vec_file_path: 预训练好的关系向量文件路径(str,path)
		:param data_file_path: 要做成graph的数据文件路径(str,path)
		:return: geometric的Data对象
		"""
		# get entitiys_vecs_dict
		entity_vec_list = open(entity_vec_file_path).readlines()
		entitys = [entity_vec.split(" ")[0] for entity_vec in entity_vec_list]
		vecs_str = [entity_vec.split(" ")[1:] for entity_vec in entity_vec_list]
		vecs = []
		for vec in vecs_str:
			vec_list = list(map(float, vec))
			vecs.append(vec_list)
		entitys_vecs_dict = {}
		for entity, vec in zip(entitys, vecs):
			entitys_vecs_dict[entity] = vec

		# get relations_vecs_dict
		relation_vec_list = open(relation_vec_file_path).readlines()
		relations = [relation_vec.split(" ")[0] for relation_vec in relation_vec_list]
		vecs_str = [relation_vec.split(" ")[1:] for relation_vec in relation_vec_list]
		vecs = []
		for vec in vecs_str:
			vec_list = list(map(float, vec))
			vecs.append(vec_list)
		relations_vecs_dict = {}
		for relation, vec in zip(relations, vecs):
			relations_vecs_dict[relation] = vec

		# get_graph
		x = []
		y = []
		#print(self.entity_id_dict)
		for entity in self.entity_id_dict.keys():
			#print(entity)
			x.append(entitys_vecs_dict[entity])
			y.append(float(self.entity_id_dict[entity]))
		x = torch.tensor(x, dtype=torch.float)
		y = torch.tensor(y, dtype=torch.float)

		if not os.path.exists("onehot.txt"):
			_,one_hot_data_list,_,_ = self.one_hot(data_file_path=data_file_path,save=True,one_hot_file_path="onehot.txt")
		else:
			one_hot_data_list = open("onehot.txt").readlines()

		h_list = [int(data.split("\t")[0]) for data in one_hot_data_list]
		l_list = [int(data.split("\t")[2][:-1]) for data in one_hot_data_list]
		r_list = [data.split("\t")[1] for data in self.data_list]

		edge_index = torch.tensor([h_list,l_list],
								  dtype=torch.long)
		edge_attr = torch.tensor([relations_vecs_dict[relation] for relation in r_list],
								 dtype=torch.float)
		graph_data = Data(x=x,y=y,edge_index=edge_index,edge_attr=edge_attr)

		return graph_data

	def get_nx_graph(self,data_file_path,show):
		"""
		:param data_file_path: 要做成graph的数据文件路径(str,path)
		:param show: 是否展示graph(bool)
		:return: nx.DiGraph对象
		"""
		if not os.path.exists("onehot.txt"):
			_,one_hot_data_list,_,_ = self.one_hot(data_file_path=data_file_path,save=True,one_hot_file_path="onehot.txt")
		else:
			one_hot_data_list = open("onehot.txt").readlines()
		graph = nx.DiGraph()
		h_t_r_list = [(int(data.split("\t")[0]),int(data.split("\t")[2][:-1]),int(data.split("\t")[1])) for data in one_hot_data_list]
		graph.add_weighted_edges_from(ebunch_to_add=h_t_r_list)
		if show:
			nx.draw(graph,node_size=5,with_labels=True)
			plt.show()
		return graph

class Evaluation:
	def __init__(self):
		pass
	def MREvaluation_CKRL(self,evalloader: dataloader,simMeasure="dot",**kwargs):
		R = 0
		N = 0
		for tri in evalloader:
			# tri : shape(N, 3)
			# head : shape(N, 1) ==> shape(N)
			# relation : shape(N, 1) ==> shape(N)
			# tail : shape(N, 1) ==> shape(N)
			tri = tri.numpy()
			head, relation, tail = tri[:, 0], tri[:, 1], tri[:, 2]
			ranks = self.evalCKRL(head, relation, tail, simMeasure, **kwargs)
			R += np.sum(ranks)
			N += ranks.shape[0]
		return R / N

	def evalCKRL(self,head, relation, tail, simMeasure, **kwargs):
		# Use np.take() to gather embedding
		head = np.take(kwargs["entityEmbed"], indices=head, axis=0)
		relation = np.take(kwargs["relationEmbed"], indices=relation, axis=0)
		# tail = np.take(kwargs["entityEmbed"], indices=tail, axis=0)
		# Calculate the similarity, sort the score and get the rank
		simScore = self.calSimilarity(head + relation, kwargs["entityEmbed"], simMeasure=simMeasure)  #||head+relation- l_embedded||
		ranks = self.calRank(simScore, tail, simMeasure=simMeasure)
		return ranks

	'''
	Used to calculate the similarity between expected tail vector and real one.
	==> expTailMatrix: shape(N, embedDim)Calculate by head and relation, 
	                   N :batchsize
	==> tailEmbedding: shape(entityNum, embedDim)The entity embedding matrix, 
	                   entityNum is the number of entities.
	==> return: shape(N, entityNum)The similarity between each vector in expTailMatrix 
	            and all vectors in tailEmbedding.
	'''

	def L2_calSimilarity(self,expTailMatrix: np.ndarray, tailEmbedding: np.ndarray):
		simScore = []
		for expM in expTailMatrix:
			# expM :          (E, ) -> (1, E)
			# tailEmbedding : (N, E)
			score = np.linalg.norm(expM[np.newaxis, :] - tailEmbedding, ord=2, axis=1, keepdims=False)
			simScore.append(score)
		return np.array(simScore)

	def calRank(self,simScore: np.ndarray, tail: np.ndarray, simMeasure: str):
		realScore = simScore[np.arange(tail.shape[0]), tail].reshape((-1, 1))
		judMatrix = simScore - realScore
		judMatrix[judMatrix > 0] = 0
		judMatrix[judMatrix < 0] = 1
		judMatrix = np.sum(judMatrix, axis=1)
		return judMatrix

if __name__ == "__main__":
	data_utiles = Data_utils(entity_id_file_path="FB15k-237/entity2id.txt",
							 relation_id_file_path="FB15k-237/relation2id.txt")
	#data_utiles.get_nx_graph(data_file_path="FB15k-237/divided dataset/train.txt",
							 #show=True)


	data_utiles.one_hot(data_file_path="FB15k-237/few simple dataset/train.txt",
						save=True,
						one_hot_file_path="onehot.txt")

	'''
	data_utiles.get_geometric_graph(entity_vec_file_path="FB15k-237/FB15K_TransE_Entity2Vec_100.txt",
									relation_vec_file_path="FB15k-237/FB15K_TransE_Relation2Vec_100.txt",
									data_file_path="FB15k-237/divided dataset/train.txt")
	'''
