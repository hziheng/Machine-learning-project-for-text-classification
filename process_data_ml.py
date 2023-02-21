#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : process_data.py
@Time    : 2023/01/13 16:25:15
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : process data, 两件事：将label和特征分开，对label做好映射和转换；采样， 解决不平衡的问题
'''

import pandas as pd
from sklearn.utils import shuffle
from collections import OrderedDict
from imblearn.under_sampling import RandomUnderSampler

class ML_Data_Excuter:
    def __init__(self, data_path, split_size, is_sample=False, split=True, train_data_path='', test_data_path=''):
        """数据处理类

        Args:
            data_path (str): 数据的路径
            split_size (int): 切分训练集和测试集的比例
            is_sample (bool, optional): 是否对数据进行采样，当数据不平衡时推荐True. Defaults to False.
            split (bool, optional): 是否进行训练集和测试集的切分操作. Defaults to True.
            train_data_path (str, optional): 如果这个路径存在，那么默认不进行程序默认的训练集和测试集的切分，用用户已经切分好的数据. Defaults to ''.
            test_data_path (str, optional): 同上. Defaults to ''.
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        if self.test_data_path and self.test_data_path:
            self.train_data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            self.data = pd.concat([self.train_data, self.test_data], axis=0)
            self.l2i_dic, self.i2l_dic = self.create_l2i()
            self.label = self.data['label']
            if len(self.label) > 2:
                self.mutil = True
            else:
                self.mutil = False
            self.X = self.data.loc[:, self.data.columns!='label']
            print('data nums: ')
            print(self.X.shape[0])
            self.train_data_label = self.train_data['label']
            self.train_data_x = self.train_data.loc[:, self.train_data.columns!='label']
            self.test_data_label = self.test_data['label']
            self.test_data_x = self.test_data.loc[:, self.test_data.columns!='label']
            print('split train_test data:')
            print('train_data num:')
            print(self.train_data_x.shape)
            print('test_data num:')
            print(self.test_data_x.shape)
        else:
            self.split_size = split_size
            self.data = pd.read_csv(data_path)
            self.l2i_dic, self.i2l_dic = self.create_l2i()
            self.label = self.data['label']
            self.X = self.data.loc[:, self.data.columns!='label']
            if is_sample:
                self.sample()
            print('data nums: ')
            print(self.X.shape[0])
            if split:
                self.train_test_split()

    def create_l2i(self):
        i2l_dic = OrderedDict()
        l2i_dic = OrderedDict()
        # 将label转成数字，并且生成有序字典，方便后续画confusion_matrix
        classes = list(set(self.data['label'].values.tolist()))
        num_classes = len(set(classes))
        for i in range(num_classes):
            i2l_dic[i] = classes[i]
            l2i_dic[classes[i]] = i
        self.data['label'] = self.data['label'].map(l2i_dic)

        return l2i_dic, i2l_dic
    
    def sample(self):
        # 这里采用简单的随机下采样，换方法可以在这里改
        def get_res():
            res = sorted(Counter(self.label).items())
            res_ = []
            for i in res:
                tmp = (self.i2l_dic[i[0]], i[1])
                res_.append(tmp)
            return res_
        from collections import Counter

        print('采样前，label的分布如下：')
        print(get_res())
        sample_excuter = RandomUnderSampler(random_state=96)
        self.X, self.label = sample_excuter.fit_resample(self.X, self.label)
        print('采样后，label的分布如下：')
        print(get_res())
        self.data = pd.concat([self.X, self.label], axis=1)

    def train_test_split(self):
        """
        这里的划分是按照每个标签的数量进行划分，确保训练集和验证集中的标签种类一致，不会出现训练集里有的标签，而测试集里没有出现过
        """
        type_label = list(set(self.data.label.values.tolist()))
        test_data_index = []
        for l in type_label:
            tmp_data = self.data[self.data['label']==l]
            tmp_data = shuffle(tmp_data)
            random_test = tmp_data.sample(frac=self.split_size, random_state=96)
            index_num = random_test.index.tolist()
            test_data_index += index_num
        test_data = self.data.iloc[test_data_index, :]
        train_data = self.data[~self.data.index.isin(test_data_index)]
        self.train_data_label = train_data['label']
        self.train_data_x = train_data.loc[:, train_data.columns!='label']
        self.test_data_label = test_data['label']
        self.test_data_x = test_data.loc[:, test_data.columns!='label']
        print('split train_test data:')
        print('train_data num:')
        print(self.train_data_x.shape)
        print('test_data num:')
        print(self.test_data_x.shape)


if __name__ == '__main__':
    data_path = './data/processed_data.csv'
    data_ex = ML_Data_Excuter(data_path, 0.3, is_sample=True, split=True)
    print(1)