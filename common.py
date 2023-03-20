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
import os
import glob
import torch
from datetime import timedelta

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def check_args(args):
    args.setting_file = os.path.join(args.checkpoint_dir, args.setting_file)
    args.log_file = os.path.join(args.checkpoint_dir, args.log_file)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(args.setting_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print('------------ Options -------------')
        for k in args.__dict__:
            v = args.__dict__[k]
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print('%s: %s' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print('------------ End -------------')

    return args


def torch_show_all_params(model, rank=0):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    if rank == 0:
        print("Total param num：" + str(k))


def torch_init_model(model, init_checkpoint, delete_module=False):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    state_dict_new = {}
    # delete module代表是否你是用了DistributeDataParallel分布式训练.
    # 这里如果是用了pytorch的DDP方式训练，要删掉module这个字段的内容
    if delete_module:
        for key in state_dict.keys():
            v = state_dict[key]
            state_dict_new[key.replace('module.', '')] = v
        state_dict = state_dict_new
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')


def torch_save_model(model, output_dir, scores, max_save_num=1):
    # Save model checkpoint
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    saved_pths = glob(os.path.join(output_dir, '*.pth'))
    saved_pths.sort()
    while len(saved_pths) >= max_save_num:
        if os.path.exists(saved_pths[0].replace('//', '/')):
            os.remove(saved_pths[0].replace('//', '/'))
            del saved_pths[0]

    save_prex = "checkpoint_score"
    for k in scores:
        save_prex += ('_' + k + '-' + str(scores[k])[:6])
    save_prex += '.pth'

    torch.save(model_to_save.state_dict(),
               os.path.join(output_dir, save_prex))
    print("Saving model checkpoint to %s", output_dir)