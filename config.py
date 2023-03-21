#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : config.py
@Time    : 2023/01/13 16:19:42
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : None
'''

ML_MODEL_NAME = ['lg', 'knn', 'dt', 'rf', 'gbdt', 'xgb', 'catboost', 'svm', 'bayes']

DL_MODEL_NAME = ['lstm', 'cnn', 'transformer', 'capsules']

PRE_MODEL_NAME = ['mac_bert', 'bert_wwm', 'bert', 'nezha_wwm', 'roberta_wwm']

BATCH_SIZE = 8

SPLIT_SIZE = 0.3

IS_SAMPLE = True

PIC_SAVED_PATH = './pic/' # result的pic图片保存的路径

VOCAB_MAX_SIZE = 100000 # 词表中词的最大数量

WORD_MIN_FREQ = 5 # 词表中一个单词出现的最小频率

VOCAB_SAVE_PATH = './data/vocab_dic.pkl' # 词表存储的位置

L2I_SAVE_PATH = './data/label2id.pkl' #  label的映射表

PRETRAIN_EMBEDDING_FILE = './data/embed.txt'

VERBOSE = 1 # 每隔10个epoch 输出一次训练结果和测试的loss

MAX_SEQ_LEN = 100 # 使用预训练模型时，设置允许每条文本数据的最长长度