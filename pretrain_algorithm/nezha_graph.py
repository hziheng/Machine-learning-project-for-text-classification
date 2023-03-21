# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Author       : Huang zh
 Email        : jacob.hzh@qq.com
 Date         : 2022-09-16 14:39:21
 LastEditTime : 2023-03-21 19:12:37
 FilePath     : \\codes\\pretrain_algorithm\\nezha_graph.py
 Description  : 
'''

import torch
from torch import nn
from transformers import NezhaPreTrainedModel, NezhaModel

class nezha_classify(NezhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config, )

        self.num_labels = config.num_labels

        self.nezha = NezhaModel(config)

        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3

        self.layer_weights = torch.nn.Parameter(weights_init)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            label=None,
    ):
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

        hidden_layers = outputs[2]

        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
        )

        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)

        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )

        return logits  
