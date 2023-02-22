#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : ml_model.py
@Time    : 2023/01/13 16:26:41
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : lg, knn, dt, rt, gbdt, xgb, catboost, svm  ...   etc.
'''

import pickle
import os
import catboost as cb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from config import ML_MODEL_NAME


class ML_EXCUTER:
    def __init__(self, model_name):
        self.model_name = model_name

    def judge_model(self):
        if self.model_name not in ML_MODEL_NAME:
            print('ml model name is not support, please see ML_MODEL_NAME of config.py')

        if self.model_name == 'lg':
            model = LogisticRegression(random_state=96)
        elif self.model_name == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
        elif self.model_name == 'bayes':
            model = GaussianNB()
        elif self.model_name == 'svm':
            model = SVC(kernel='rbf')
        elif self.model_name == 'dt':
            model = DecisionTreeClassifier(random_state=96)
        elif self.model_name == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=96)
        elif self.model_name == 'gbdt':
            model = GradientBoostingClassifier(
                learning_rate=0.1, n_estimators=100, random_state=96)
        elif self.model_name == 'xgb':
            model = XGBClassifier(learning_rate=0.1,
                                  # n_estimatores
                                  # 含义：总共迭代的次数，即决策树的个数
                                  n_estimators=1000,
                                  # max_depth
                                  # 含义：树的深度，默认值为6，典型值3-10。
                                  max_depth=6,
                                  # min_child_weight
                                  # 调参：值越大，越容易欠拟合；值越小，越容易过拟合
                                  # （值较大时，避免模型学习到局部的特殊样本）。
                                  min_child_weight=1,
                                  # 惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                                  gamma=0,
                                  # subsample
                                  # 含义：训练每棵树时，使用的数据占全部训练集的比例。
                                  # 默认值为1，典型值为0.5-1。
                                  subsample=0.8,
                                  # colsample_bytree
                                  # 含义：训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。
                                  colsample_btree=0.8,
                                  # objective 目标函数
                                  # multi：softmax num_class=n 返回类别
                                  # binary:logistic,二元分类的逻辑回归，输出概率 binary:hinge：二进制分类的铰链损失。这使预测为0或1，而不是产生概率。
                                  objective='multi:softmax',
                                  num_class=3,
                                  # scale_pos_weight
                                  # 正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时，scale_pos_weight=10
                                  scale_pos_weight=1,
                                  random_state=96
                                  )
            # xgb 的调参看这篇文章：https://zhuanlan.zhihu.com/p/143009353
        elif self.model_name == 'catboost':
            # 详细调参和gpu训练看这里：http://t.zoukankan.com/webRobot-p-9249906.html
            model = cb.CatBoostClassifier(iterations=500,
                                          learning_rate=0.1,
                                          max_depth=6,
                                          verbose=100,
                                          early_stopping_rounds=500,
                                          loss_function='Logloss',
                                          task_type='CPU',  # 'GPU'
                                          random_seed=96,
                                          one_hot_max_size=2
                                          )

        else:
            pass
        self.model = model

    def train(self, x_data, y_data):
        self.model.fit(x_data, y_data)

    def predict(self, data):
        return self.model.predict(data)

    def save_model(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        output_path = os.path.join(path, name)
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f'model is saved, in {str(output_path)}')

    def load_model(self, path, name):
        output_path = os.path.join(path, name)
        try:
            with open(output_path, 'rb') as f:
                self.model = pickle.load(f)
            print('model is load')
        except:
            print('model load fail, check path')
