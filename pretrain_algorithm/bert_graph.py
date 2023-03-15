# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Author       : Huang zh
 Email        : jacob.hzh@qq.com
 Date         : 2023-03-12 14:39:21
 LastEditTime : 2023-03-14 19:20:06
 FilePath     : \\codes\\pretrain_algorithm\\bert_graph.py
 Description  : 
'''


import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class bert_classifier(BertPreTrainedModel):
    '''
    pooler_output：shape是(batch_size, hidden_size)，这是序列的第一个token (cls) 的最后一层的隐藏状态，
    它是由线性层和Tanh激活函数进一步处理的，
    这个输出不是对输入的语义内容的一个很好的总结，对于整个输入序列的隐藏状态序列的平均化或池化可以更好的表示一句话。(这里还加入了embedding层和
    每个隐藏层的cls进行加权平均化来表示一句话)
    '''

    def __init__(self, config):
        super().__init__(config, )
        config.output_hidden_states = True
        '''
        hidden_states：这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它是一个元组，含有13个元素，
        第一个元素可以当做是embedding，也就是cls，其余12个元素是各层隐藏状态的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)，
        '''
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)
        n_weights = config.num_hidden_layers + 1  # 因为指定了输出hidden_states，所以多了一层，加1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, label=None,):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        '''
        bert的输出
        # output[0] 最后一层的隐藏状态 （batch_size, sequence_length, hidden_size)
        # output[1] 第一个token即（cls）最后一层的隐藏状态 (batch_size, hidden_size)
        # output[2] 需要指定 output_hidden_states = True， 包含所有隐藏状态，第一个元素是embedding, 其余元素是各层的输出 (batch_size, sequence_length, hidden_size)
        # output[3] 需要指定output_attentions=True，包含每一层的注意力权重，用于计算self-attention heads的加权平均值(batch_size, layer_nums, sequence_length, sequence_legth)
        '''
        hidden_layers = outputs[2]
        # 取每一层的cls（shape：batchsize * hidden_size) dropout叠加 shape: 13*bathsize*hidden_size
        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=0
        )
        # 然后加权求和 shape: bathsize*hidden_size
        cls_output = (torch.softmax(self.layer_weights,
                      dim=0).unsqueeze(-1).unsqueeze(-1) * cls_outputs).sum(0)
        # 对求和后的cls向量进行dropout，在输入线性层，重复五次，然后求平均的到最后的输出logit
        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output))
                 for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )

        return logits
