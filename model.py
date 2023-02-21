#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : model.py
@Time    : 2023/02/07 19:54:07
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : None
'''

from config import ML_MODEL_NAME, DL_MODEL_NAME
from ml_algorithm.ml_model import ML_EXCUTER
from dl_algorithm.dl_model import DL_EXCUTER

class Model_Excuter:
    def __init__(self):
        pass
    # def init(self, model_name):
    #     if model_name in ML_MODEL_NAME:
    #         return ML_EXCUTER(model_name)
    #     elif model_name in DL_MODEL_NAME:
    #         return DL_EXCUTER(model_name)
    def init(self, model_name='', dl_config=''):
        if model_name in ML_MODEL_NAME:
            return ML_EXCUTER(model_name)
        elif dl_config.model_name in DL_MODEL_NAME:
            return DL_EXCUTER(dl_config)