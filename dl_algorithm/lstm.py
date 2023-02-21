#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : lstm.py
@Time    : 2023/02/07 16:09:05
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : lstm for classifier
'''

import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, dlconfig):
        super().__init__()
        if dlconfig.embedding_pretrained == 'random':
            self.embedding = nn.Embedding(dlconfig.vocab_size, dlconfig.embedding_size, padding_idx=dlconfig.vocab_size-1)
        else:
            self.embedding = nn.Embedding.from_pretrained(dlconfig.embedding_matrix, freeze=False, padding_idx=dlconfig.vocab_size-1)
        self.lstm = nn.LSTM(dlconfig.embedding_size, dlconfig.hidden_size, batch_first=True, bidirectional=True, dropout=dlconfig.dropout)
        self.fc1 = nn.Linear(dlconfig.hidden_size*2, dlconfig.hidden_size)
        self.fc2 = nn.Linear(dlconfig.hidden_size, dlconfig.nums_label)
        self.dropout = nn.Dropout(p=dlconfig.dropout)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.embedding(x) # [batch_size, seq_len, embeding_size]
        x,_ = self.lstm(x)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x[:, -1, :]

        