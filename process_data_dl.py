#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : utils.py
@Time    : 2023/02/08 14:57:32
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : get vocab, label, label_nums, label2n, word2n, n2word, n2label, dataset定义
'''

import os
import jieba
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from config import VOCAB_MAX_SIZE, WORD_MIN_FREQ, VOCAB_SAVE_PATH, L2I_SAVE_PATH, PRETRAIN_EMBEDDING_FILE
from trick.dynamic_padding import collater


class DataSetProcess:
    def __init__(self, train_data_path='', test_data_path='', dev_data_path=''):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.dev_data_path = dev_data_path
        self.train_data, self.l1 = self.load_data(
            self.train_data_path) if self.train_data_path else [[], []]
        self.test_data, self.l2 = self.load_data(
            self.test_data_path) if self.test_data_path else [[], []]
        self.dev_data, self.l3 = self.load_data(
            self.dev_data_path) if self.dev_data_path else [[], []]

    def load_data(self, path):
        """默认是处理csv文件，其他形式的要改，csv的话文本内容要改成我提供的demo格式
        """
        if path.endswith('csv'):
            df = pd.read_csv(path, encoding='utf-8')
            contents = df['content'].values.tolist()
            try:
                labels = df['label'].values.tolist()
            except:
                labels = []
            return contents, labels
        else:
            #! todo 其他格式的文件读取
            pass

    def build_vocab(self, save=False):
        if os.path.exists(VOCAB_SAVE_PATH):
            with open(VOCAB_SAVE_PATH, 'rb') as f:
                vocab_dic = pkl.load(f)
            print(f"vocab size {len(vocab_dic)}")
            return vocab_dic

        vocab_dic = {}
        UNK, PAD = '<UNK>', '<PAD>'
        min_freq = WORD_MIN_FREQ
        vocab_max_size = VOCAB_MAX_SIZE

        all_data = self.train_data + self.test_data + self.dev_data

        for sentence in all_data:
            sentence = sentence.strip()
            #! 这里只设置了中文，英文用空格，还没写
            tokens = jieba.cut(sentence)
            for token in tokens:
                vocab_dic[token] = vocab_dic.get(token, 0) + 1
        # 对词表进行排序
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[
                            1] >= min_freq], key=lambda x: x[1], reverse=True)[:vocab_max_size]

        # 还原成字典
        vocab_dic = {word_count[0]: idx for idx,
                     word_count in enumerate(vocab_list)}

        # 使用UNK填充单词表的尾部
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

        # 是否保存
        if save:
            abs_path = VOCAB_SAVE_PATH.rsplit('/', 1)[0]
            if not os.path.exists(abs_path):
                os.makedirs(abs_path)
            with open(VOCAB_SAVE_PATH, 'wb') as f:
                pkl.dump(vocab_dic, f)
            print(f'vocab_dic is saved in {VOCAB_SAVE_PATH}')
        print(f"vocab size {len(vocab_dic)}")
        return vocab_dic

    def build_label2id(self, save=False):
        if os.path.exists(L2I_SAVE_PATH):
            with open(L2I_SAVE_PATH, 'rb') as f:
                l2i_dic = pkl.load(f)
            i2l_dic = {}
            for k, n in l2i_dic.items():
                i2l_dic[n] = k
            return l2i_dic, i2l_dic


        i2l_dic = OrderedDict()
        l2i_dic = OrderedDict()
        all_label_list = self.l1 + self.l2 + self.l3
        all_label_list = list(set(all_label_list))
        for i in range(len(all_label_list)):
            i2l_dic[i] = all_label_list[i]
            l2i_dic[all_label_list[i]] = i

        # 是否保存
        if save:
            abs_path = L2I_SAVE_PATH.rsplit('/', 1)[0]
            if not os.path.exists(abs_path):
                os.makedirs(abs_path)
            with open(L2I_SAVE_PATH, 'wb') as f:
                pkl.dump(l2i_dic, f)
            print(f'label2id_dic is saved in {L2I_SAVE_PATH}')

        return l2i_dic, i2l_dic

    def trans_data(self, data_path, vocab_dic, label_dic):
        contents = []
        datas, labels = self.load_data(data_path)
        if not labels:
            labels = [-1] * len(datas)
        for d, l in zip(datas, labels):
            if not d.strip():
                continue
            wordlists = []
            tokens = jieba.cut(d.strip())
            for token in tokens:
                wordlists.append(vocab_dic.get(token, vocab_dic.get("<UNK>")))
            if l != -1:
                contents.append((wordlists, int(label_dic.get(l))))
            else:
                contents.append((wordlists,))
        return contents

    def load_emb(self, vocab_dic):
        skip = True
        emb_dic = {}
        with open(PRETRAIN_EMBEDDING_FILE, 'r', encoding='utf-8') as f:
            for i in f:
                if skip:
                    skip = False
                    dim = int(i.split(' ', 1)[1].strip())
                    continue
                word, embeds = i.split(' ', 1)
                embeds = embeds.strip().split(' ')
                if vocab_dic.get(word, None):
                    emb_dic[vocab_dic[word]] = embeds
        emb_dics = sorted(emb_dic.items(), key=lambda x: x[0])
        orignal_emb = nn.Embedding(len(vocab_dic), dim, padding_idx=len(vocab_dic)-1)
        emb_array = orignal_emb.weight.data.numpy()
        for i in emb_dics:
            index = i[0]
            weight = np.array(i[1], dtype=float)
            emb_array[index] = weight
        print(f'已载入预训练词向量，维度为{dim}')
        return torch.FloatTensor(emb_array), dim


class DLDataset(Dataset):
    """自定义torch的dataset
    """

    def __init__(self, contents):
        self.data, self.label = self.get_data_label(contents)
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.label:
            return {
                'input_ids': self.data[index],
                'label': self.label[index]
            }
        else:
            return {'input_ids': self.data[index]}

    def get_data_label(self, contents):
        """contents: [([xx,x,,], label?), ()]
        """
        data = []
        label = []
        for i in contents:
            data.append(i[0])
            if len(i) == 2:
                label.append(i[1])
        return data, label

class DL_Data_Excuter:
    def __init__(self):
        pass
    def process(self,batch_size, train_data_path='', test_data_path='', dev_data_path=''):
        """内部构建各个数据集的dataloader，返回词表大小和类别数量
        """

        p = DataSetProcess(train_data_path, test_data_path, dev_data_path)
        self.vocab = p.build_vocab(save=True)
        pad_index = self.vocab['<PAD>']
        self.label_dic, self.i2l_dic = p.build_label2id(save=True)
        if len(self.label_dic) > 2:
            self.multi = True
        else:
            self.multi = False
        collater_fn = collater(pad_index)
        self.train_data_loader = ''
        self.test_data_loader = ''
        self.dev_data_loader = ''
        if train_data_path:
            content = p.trans_data(train_data_path, self.vocab, self.label_dic)
            data_set = DLDataset(content)
            self.train_data_loader = DataLoader(
                data_set, batch_size=batch_size, shuffle=True, collate_fn=collater_fn)
        if test_data_path:
            content = p.trans_data(test_data_path, self.vocab, self.label_dic)
            data_set = DLDataset(content)
            self.test_data_loader = DataLoader(
                data_set, batch_size=batch_size, shuffle=False, collate_fn=collater_fn)
        if dev_data_path:
            content = p.trans_data(dev_data_path, self.vocab, self.label_dic)
            data_set = DLDataset(content)
            self.dev_data_loader = DataLoader(
                data_set, batch_size=batch_size, shuffle=False, collate_fn=collater_fn)
        return len(self.vocab), len(self.label_dic)


if __name__ == '__main__':
    d = DL_Data_Excuter()
    d.get_dataloader(2, '', './data/dl_data/test.csv', '')
    print(1)
