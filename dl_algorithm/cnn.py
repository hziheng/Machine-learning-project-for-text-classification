#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : cnn.py
@Time    : 2023/01/17 15:22:31
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : cnn for textclassifier
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, dlconfig):
        super().__init__()
        if dlconfig.embedding_pretrained == 'random':
            self.embedding = nn.Embedding(dlconfig.vocab_size, dlconfig.embedding_size, padding_idx=dlconfig.vocab_size-1)
        else:
            self.embedding = nn.Embedding.from_pretrained(dlconfig.embedding_matrix, freeze=False, padding_idx=dlconfig.vocab_size-1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, dlconfig.nums_filters, (k, dlconfig.embedding_size), stride=dlconfig.stride, padding=dlconfig.pad_size) for k in dlconfig.filter_size]
        )
        self.dropout = nn.Dropout(p=dlconfig.dropout)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(dlconfig.nums_filters * len(dlconfig.filter_size), dlconfig.nums_label)

    def conv_and_pool(self, x, conv):
        x = self.relu(conv(x))
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1) # 增加通道数为1
        x = [self.conv_and_pool(x, conv) for conv in self.convs]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x