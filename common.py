#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : common.py
@Time    : 2023/02/09 14:33:09
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : some common func
'''

import time
from datetime import timedelta

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


