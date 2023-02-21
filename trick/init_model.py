#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : init_model.py
@Time    : 2023/02/09 14:26:06
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : dl net权重初始化方式
'''

import torch.nn as nn


def init_network(model, method='xavier', exclude='embedding'):
    # 权重初始化：不同的初始化方法，导致精确性和收敛时间不同
    # 默认xavier
    # xavier：“Xavier”初始化方法是一种很有效的神经网络初始化方法
    # kaiming：何凯明初始化
    # normal_: 正态分布初始化
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name and 'layernorm' not in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass