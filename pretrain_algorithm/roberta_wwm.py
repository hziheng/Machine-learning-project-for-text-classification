# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Author       : Huang zh
 Email        : jacob.hzh@qq.com
 Date         : 2023-03-21 19:14:06
 LastEditTime : 2023-03-21 19:54:49
 FilePath     : \\codes\\pretrain_algorithm\\roberta_wwm.py
 Description  : 
'''


import torch
from torch import nn
from transformers import RobertaPreTrainedModel, RobertaModel

class roberta_classify(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config):
        super().__init__(config, )

        self.num_labels = config.num_labels
        # 如果add_pooling_layer设置为True，那么output会多一个池化层结果，可以选择用这个池化层的结果去做下游任务
        # 由于这里用多个隐层的平均作为下游任务的输入，所以设置为Fasle
        self.roberta = RobertaModel(config, add_pooling_layer=False)

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
        outputs = self.roberta(
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
