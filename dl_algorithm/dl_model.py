#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : dl_model.py
@Time    : 2023/02/07 19:54:48
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : deep net model excuter
'''

import os
import time
import torch
import gc
import numpy as np
from tqdm import tqdm
from metrics import Matrix
from config import DL_MODEL_NAME, VERBOSE
from dl_algorithm.lstm import LSTM
from dl_algorithm.cnn import TextCNN
from dl_algorithm.transformer import TransformerModel
from dl_algorithm.capsules_model import capsules_model
from trick.init_model import init_network
from trick.early_stop import EarlyStopping
from common import get_time_dif


class DL_EXCUTER:
    def __init__(self, dl_config):
        self.dlconfig = dl_config

    def judge_model(self):
        if self.dlconfig.model_name not in DL_MODEL_NAME:
            print('dl model name is not support, please see DL_MODEL_NAME of config.py')
        if self.dlconfig.model_name == 'lstm':
            self.model = LSTM(self.dlconfig)
        elif self.dlconfig.model_name == 'cnn':
            self.model = TextCNN(self.dlconfig)
        elif self.dlconfig.model_name == 'transformer':
            self.model = TransformerModel(self.dlconfig)
        elif self.dlconfig.model_name == 'capsules':
            self.model = capsules_model(self.dlconfig)
        #! 其他模型
        else:
            pass
        init_network(self.model)
        print('初始化网络权重完成，默认采用xavier')
        self.model.to(self.dlconfig.device)
        

    def train(self, train_loader, test_loader, dev_loader, model_saved_path, model_name):
        # 设置优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.dlconfig.learning_rate)
        best_test_f1 = 0
        # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        # 学习更新策略--预热(warmup)
        if self.dlconfig.update_lr:
            from transformers import get_linear_schedule_with_warmup
            num_warmup_steps = int(self.dlconfig.warmup_prop * self.dlconfig.epochs * len(train_loader))
            num_training_steps = int(self.dlconfig.epochs * len(train_loader))
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        # 早停策略
        early_stopping = EarlyStopping(patience = 20, delta=0)  

        for epoch in range(self.dlconfig.epochs):
            # 设定训练模式
            self.model.train()
            # 梯度清零
            self.model.zero_grad()
            start_time = time.time()
            avg_loss = 0
            first_epoch_eval = 0
            for data in tqdm(train_loader, ncols=100):
                pred = self.model(data['input_ids'].to(self.dlconfig.device))
                loss = self.dlconfig.loss_fct(pred, data['label'].to(self.dlconfig.device)).mean()
                # 反向传播
                loss.backward()
                avg_loss += loss.item() / len(train_loader)

                # 更新优化器
                optimizer.step()
                # 更新学习率
                if self.dlconfig.update_lr:
                    scheduler.step()

                # 用以下方式替代model.zero_grad()，可以提高gpu利用率
                for param in self.model.parameters():
                    param.grad = None
            

            
            # 计算模型运行时间
            elapsed_time = get_time_dif(start_time)
            # 打印间隔
            if (epoch + 1) % VERBOSE == 0:
                # 在测试集上看下效果
                avg_test_loss, test_f1, pred_all, true_all = self.evaluate(test_loader)
                elapsed_time = elapsed_time * VERBOSE
                if self.dlconfig.update_lr:
                    lr = scheduler.get_last_lr()[0] 
                else:
                    lr = self.dlconfig.learning_rate
                tqdm.write(
                    f"Epoch {epoch + 1:02d}/{self.dlconfig.epochs:02d} \t t={elapsed_time} \t"
                    f"loss={avg_loss:.3f}\t lr={lr:.1e}",
                    end="\t",
                )

                if (epoch + 1 >= first_epoch_eval) or (epoch + 1 == self.dlconfig.epochs):
                    tqdm.write(f"val_loss={avg_test_loss:.3f}\ttest_f1={test_f1:.4f}\t lr={lr:.1e}")
                else:
                    tqdm.write("")
            
            # 每次保存最优的模型，以测试集f1为准
            if best_test_f1 < test_f1:
                best_test_f1 = test_f1
                tqdm.write('*'*20)
                self.save_model(model_saved_path, model_name)
                tqdm.write('new model saved')
                tqdm.write('*'*20)
            
            early_stopping(avg_test_loss)
            if early_stopping.early_stop:
                break
        # 删除数据加载器以及变量
        del (test_loader, train_loader, loss, data, pred)
        # 释放内存
        gc.collect()
        torch.cuda.empty_cache()


    def evaluate(self, test_loader):
        pre_all = []
        true_all = []
        # 设定评估模式
        self.model.eval()
        avg_test_loss = 0
        with torch.no_grad():
            for test_data in test_loader:
                pred = self.model(test_data['input_ids'].to(self.dlconfig.device))
                test_loss = self.dlconfig.loss_fct(pred, test_data['label'].to(self.dlconfig.device)).mean()
                avg_test_loss += test_loss.item() / len(test_loader)
                true_all.extend(test_data['label'].detach().cpu().numpy())
                pre_all.append(pred.softmax(-1).detach().cpu().numpy())
        pre_all = np.concatenate(pre_all)
        pre_all = np.argmax(pre_all, axis=-1)
        if self.dlconfig.loss_type == 'multi' or self.dlconfig.loss_type == 'marginLoss':
            multi = True
        else:
            multi = False
        matrix = Matrix(true_all, pre_all, multi=multi)
        return avg_test_loss, matrix.get_f1(), pre_all, true_all


    def predict(self, dev_loader):
        pre_all = []
        with torch.no_grad():
            for test_data in dev_loader:
                pred = self.model(test_data['input_ids'].to(self.dlconfig.device))
                pre_all.append(pred.softmax(-1).detach().cpu().numpy())
        pre_all = np.concatenate(pre_all)
        pre_all = np.argmax(pre_all, axis=-1)
        return pre_all

    # 保存模型权重
    def save_model(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        output_path = os.path.join(path, name)
        torch.save(self.model.state_dict(), output_path)
        print(f'model is saved, in {str(output_path)}')

    def load_model(self, path, name):
        output_path = os.path.join(path, name)
        try:
            self.judge_model()
            self.model.load_state_dict(torch.load(output_path))
            self.model.eval()
            print('model 已加载预训练参数')
        except:
            print('model load error')

    

