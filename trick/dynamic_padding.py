#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : dynamic_padding.py
@Time    : 2023/02/09 10:17:47
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : 每个batch保持一个长度,而不是所有的数据保持一个长度
'''

import torch

class collater():
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        # dynamic_pad
        input_ids, label = [], []
        collate_max_len = 0
        
        # get maxlen for a batch
        for data in batch:
            collate_max_len = max(collate_max_len, len(data['input_ids']))
        
        for data in batch:
            # padding to maxlen for each data
            length = len(data['input_ids'])
            input_ids.append(data['input_ids'] + [self.pad_index] * (collate_max_len - length))
            if len(data) >= 2:
                label.append(data['label'])
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        if label:
            label = torch.tensor(label, dtype=torch.long)
            return {
                'input_ids': input_ids,
                'label': label
            }
        else:
            return {'input_ids': input_ids}

