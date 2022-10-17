# 王冬虎
# 学习时间:2022/10/11 14:32
import os
import json
import torch

config_json_date = open("path.json", "r", encoding="gb2312")
json_config = json.loads(config_json_date.read())
base_path_dict = json_config["base_path"]
load_path_dict = json_config["load_path_dict"]

class TrainConfig:
	def __init__(self):
		#path arguments
		self.entity_id_file_path = base_path_dict["entity_id_file_path"]
		self.relation_id_file_path = base_path_dict["relation_id_file_path"]
		self.one_hot_data_file_path = base_path_dict["one_hot_data_file_path"]
		self.train_data_file_path = base_path_dict["train_data_file_path"]
		self.valid_data_file_path = base_path_dict["valid_data_file_path"]
		self.entity_vec_file_path = base_path_dict["entity_vec_file_path"]
		self.relation_vec_file_path = base_path_dict["relation_vec_file_path"]
		self.modelpath = load_path_dict["model"]
		self.summarydir = load_path_dict["summary_dir"]

		# Dataloader arguments
		self.batchsize = 9
		self.shuffle = True
		self.numworkers = 0
		self.droplast = False
		self.repproba = 0.5
		self.exproba = 0.5
		# Model and training general arguments
		self.modelname = "CKRL"
		self.CKRL = {
			"EmbeddingDim": 100,
			"Margin": 1.0,
			"L": 2}
		self.usegpu = torch.cuda.is_available()
		self.premodel = False
		self.weightdecay = 0

		self.epochs = 5
		self.evalepoch = 1
		self.learningrate = 0.01
		self.lrdecay = 0.96
		self.lrdecayepoch = 5
		self.optimizer = "Adam"
		# Evaluation arguments
		self.evalmethod = "MREvaluation_CKRL"
		self.simmeasure = "L2"
		# Save model arguments
		self.modelsave = "param"




	def print_args(self):
		print("="*20+"Arguments"+"="*20)
		argsDict = vars(self)
		for arg, value in argsDict.items():
			print("==> {} : {}".format(arg, value))
		print("=" * 50)