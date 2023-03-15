# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Author       : Huang zh
 Email        : jacob.hzh@qq.com
 Date         : 2023-03-13 17:10:12
 LastEditTime : 2023-03-14 20:05:58
 FilePath     : \\codes\\pretrain_algorithm\\pre_model.py
 Description  : 
'''

import gc
import os
import shutil
import numpy as np
import torch
import time
from tqdm import tqdm
from common import get_time_dif
from config import PRE_MODEL_NAME, VERBOSE
from metrics import Matrix
from pretrain_algorithm.bert_graph import bert_classifier
from transformers import BertConfig
from trick.early_stop import EarlyStopping
from trick.fgm_pgd_ema import FGM


class PRE_EXCUTER:
    def __init__(self, dl_config):
        self.dlconfig = dl_config

    def judge_model(self, assign_path=''):
        load_path = assign_path
        if self.dlconfig.model_name not in PRE_MODEL_NAME:
            print('pretrain model name is not support, please see PRE_MODEL_NAME of config.py')
        if self.dlconfig.model_name == 'bert':
            self.pre_config = BertConfig.from_pretrained(os.path.join(load_path, 'config.json'))
            self.pre_config.num_labels = self.dlconfig.nums_label
            self.model = bert_classifier.from_pretrained(os.path.join(
                load_path, 'pytorch_model.bin'), config=self.pre_config)
        #! 其他模型
        else:
            pass
        self.model.to(self.dlconfig.device)

    def train(self, train_loader, test_loader, dev_loader, model_saved_path):
        # 设置优化器
        # 带这些名字的参数不需要做权重衰减
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, 'lr': self.dlconfig.learning_rate},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': self.dlconfig.learning_rate},
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.dlconfig.learning_rate)
        best_test_f1 = 0
        # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        # 学习更新策略--预热(warmup)
        if self.dlconfig.update_lr:
            from transformers import get_linear_schedule_with_warmup
            num_warmup_steps = int(
                self.dlconfig.warmup_prop * self.dlconfig.epochs * len(train_loader))
            num_training_steps = int(self.dlconfig.epochs * len(train_loader))
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps)

        # 早停策略
        early_stopping = EarlyStopping(patience=20, delta=0)

        for epoch in range(self.dlconfig.epochs):
            # 设定训练模式
            self.model.train()
            # 梯度清零
            self.model.zero_grad()
            start_time = time.time()
            avg_loss = 0
            first_epoch_eval = 0
            for data in tqdm(train_loader, ncols=100):
                data['input_ids'] = data['input_ids'].to(self.dlconfig.device)
                data['attention_mask'] = data['attention_mask'].to(self.dlconfig.device)
                data['token_type_ids'] = data['token_type_ids'].to(self.dlconfig.device)
                data['label'] = data['label'].to(self.dlconfig.device)
                pred = self.model(**data)
                loss = self.dlconfig.loss_fct(pred, data['label']).mean()
                # 反向传播
                loss.backward()
                avg_loss += loss.item() / len(train_loader)

                # 使用fgm
                if self.dlconfig.use_fgm:
                    fgm = FGM(self.model)
                    fgm.attack()
                    loss_adv = self.model(**data).mean()
                    # 通过扰乱后的embedding训练后得到对抗训练后的loss值，然后反向传播计算对抗后的梯度，累加到前面正常的梯度上，最后再去更新参数
                    loss_adv.backward()
                    fgm.restore()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 对应上面的梯度衰减

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
                avg_test_loss, test_f1, pred_all, true_all = self.evaluate(
                    test_loader)
                elapsed_time = elapsed_time * VERBOSE
                if self.dlconfig.update_lr:
                    lr = scheduler.get_last_lr()[0]
                else:
                    lr = self.dlconfig.learning_rate
                tqdm.write(
                    f"Epoch {epoch + 1:02d}/{self.dlconfig.epochs:02d} \t time={elapsed_time} \t"
                    f"loss={avg_loss:.3f}\t lr={lr:.1e}",
                    end="\t",
                )

                if (epoch + 1 >= first_epoch_eval) or (epoch + 1 == self.dlconfig.epochs):
                    tqdm.write(
                        f"val_loss={avg_test_loss:.3f}\ttest_f1={test_f1:.4f}\t lr={lr:.1e}")
                else:
                    tqdm.write("")

            # 每次保存最优的模型，以测试集f1为准
            if best_test_f1 < test_f1:
                best_test_f1 = test_f1
                tqdm.write('*' * 20)
                self.save_model(model_saved_path)
                tqdm.write('new model saved')
                tqdm.write('*' * 20)

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
                pred = self.model(test_data['input_ids'].to(self.dlconfig.device),
                                  test_data['attention_mask'].to(self.dlconfig.device),
                                  test_data['token_type_ids'].to(self.dlconfig.device),
                                  )
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
                pred = self.model(test_data['input_ids'].to(self.dlconfig.device),
                                  test_data['attention_mask'].to(self.dlconfig.device),
                                  test_data['token_type_ids'].to(self.dlconfig.device),
                                  )
                pre_all.append(pred.softmax(-1).detach().cpu().numpy())
        pre_all = np.concatenate(pre_all)
        pre_all = np.argmax(pre_all, axis=-1)
        return pre_all

    # 保存模型权重
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, 'config.json')):
            shutil.copy(f'{self.dlconfig.pretrain_file_path}/config.json', f'{path}/config.json')
        if not os.path.exists(os.path.join(path, 'vocab.txt')):
            shutil.copy(f'{self.dlconfig.pretrain_file_path}/vocab.txt', f'{path}/vocab.txt')
        name = 'pytorch.bin'
        output_path = os.path.join(path, name)
        torch.save(self.model.state_dict(), output_path)
        print(f'model is saved, in {str(output_path)}')

    def load_model(self, path):
        try:
            self.model = self.judge_model(path)
            self.model.eval()
            print('model 已加载预训练参数')
        except:
            print('model load error')
