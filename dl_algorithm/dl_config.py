#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : dl_config.py
@Time    : 2023/02/07 17:27:38
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : some params of dl model, 参数全部在这里改
'''

import torch
from process_data_dl import DataSetProcess
from dl_algorithm.capsules_model import MarginLoss
from config import PRE_MODEL_NAME

class DlConfig:
    """
    model_name: LSTM, CNN, Transformer, capsules...
    """

    def __init__(self, model_name, vocab_size, label2id_nums, vocab_dict, embedding_pretrained='pretrain'):
        self.model_name = model_name
        self.train_data_path = ''
        self.test_data_path = ''
        self.dev_data_path = ''
        self.vocab_size = vocab_size
        self.nums_label = label2id_nums
        self.embedding_size = 200
        self.embedding_pretrained = embedding_pretrained # random, pretrain
        if self.embedding_pretrained != 'random':
            self.embedding_matrix, dim = DataSetProcess().load_emb(vocab_dict)
            self.embedding_size = dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dropout = 0.5
        self.epochs = 5
        self.learning_rate = 3e-5
        self.update_lr = True # 是否使用衰减学习率的方法动态更新学习率
        self.warmup_prop = 0.1 # 学习率更新策略系数
        self.loss_type = 'multi' # 'binary, regression, marginLoss, multi'
        self.judge_loss_fct()
        self.create_special_params()
    
    def create_special_params(self):
        if self.model_name == 'lstm':
            self.hidden_size = 128
            self.nums_layer = 1 # lstm的层数stack的
        elif self.model_name == 'cnn':
            self.nums_filters = 256 # 卷积核的数量
            self.filter_size = (2, 3, 4) # 相当于提取2gram，3gram，4gram的信息
            self.stride = 1
            self.pad_size = 0
        elif self.model_name == 'transformer':
            self.heads = 5 # 确保能被embed_size 整除
            self.n_layers = 2 # encoder里有几个transformer
            self.hidden = 1024
            self.d_model = self.embedding_size
        elif self.model_name == 'capsules':
            # 注意：self.in_d * self.reshape_num = 256
            self.in_d = 8
            self.reshape_num = 32
            self.out_d = 16
            self.iter = 3 # cij 的迭代次数
            self.pad_size = 0
        #==============================#
        elif self.model_name in PRE_MODEL_NAME:
            self.use_fgm = True # 是否使用fgm (Fast Gradient Method)
        else:
            pass

    def judge_loss_fct(self):
        if self.loss_type == 'multi':
            # torch.nn.CrossEntropyLoss(input, target)的input是没有归一化的每个类的得分，而不是softmax之后的分布
            # target是：类别的序号。形如 target = [1, 3, 2]
            self.loss_fct = torch.nn.CrossEntropyLoss()
        elif self.loss_type == 'binary':
            self.loss_fct = torch.nn.BCELoss()
        elif self.loss_type == 'regression':
            self.loss_fct = torch.nn.MSELoss()
        elif self.loss_type == 'marginLoss':
            self.loss_fct = MarginLoss()
        else:
            #! 这里自定义loss函数
            pass
    

