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
    def __init__(self, pad_index, for_pretrain=False):
        # 如果for_pretrain=True，说明返回的数据要包含attention矩阵和token_ids矩阵
        self.pad_index = pad_index
        self.for_pretrain = for_pretrain

    def __call__(self, batch):
        # dynamic_pad
        input_ids, label = [], []
        collate_max_len = 0
        attention_mask, token_type_ids = [], []
        
        # get maxlen for a batch
        for data in batch:
            collate_max_len = max(collate_max_len, len(data['input_ids']))
        
        for data in batch:
            # padding to maxlen for each data
            length = len(data['input_ids'])
            input_ids.append(data['input_ids'] + [self.pad_index] * (collate_max_len - length))
            if self.for_pretrain:
                attention_mask.append(data['attention_mask'] + [self.pad_index] * (collate_max_len - length))
                token_type_ids.append(data['token_type_ids'] + [self.pad_index] * (collate_max_len - length))
            if len(data) >= 2:
                label.append(data['label'])
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        result = {'input_ids': input_ids}
        if label:
            label = torch.tensor(label, dtype=torch.long)
            result['label'] = label
        if self.for_pretrain:
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            result['attention_mask'] = attention_mask
            result['token_type_ids'] = token_type_ids
        return result

