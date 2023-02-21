#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : metrics.py
@Time    : 2023/01/15 11:35:31
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : 一系列的评估函数f1, recall, acc, presion, confusion_matrix...
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import confusion_matrix
from config import PIC_SAVED_PATH


class Matrix:
    def __init__(self, y_true, y_pre, mutil=False):
        self.true = y_true
        self.pre = y_pre
        # 是否是多分类, 默认二分类
        self.mutil = mutil # average的参数有micro、macro、weighted,如果选择micro,那么recall和pre和acc没区别，建议使用macro，同时数据集最好已经没有不平衡的问题

    def get_acc(self):
        return accuracy_score(self.true, self.pre)

    def get_recall(self):
        # tp / (tp + fn)
        if self.mutil:
            return recall_score(self.true, self.pre, average='macro')
        return recall_score(self.true, self.pre)
    
    def get_precision(self):
        # tp / (tp + fp)
        if self.mutil:
            return precision_score(self.true, self.pre, average='macro')
        return precision_score(self.true, self.pre)

    def get_f1(self):
        # F1 = 2 * (precision * recall) / (precision + recall)
        if self.mutil:
            return f1_score(self.true, self.pre, average='macro')
        return f1_score(self.true, self.pre)

    def get_confusion_matrix(self):
        return confusion_matrix(self.true, self.pre)

    def plot_confusion_matrix(self, dic_labels):
        """plot

        Args:
            dic_labels (dict): {0: 'label1', 1: 'label2'} # 一定是个有序字典
        """
        proportion = []
        con_matrix = self.get_confusion_matrix()
        num_class = len(dic_labels)
        labels = [v for k, v in dic_labels.items()]
        for i in con_matrix:
            for j in i:
                temp = j / (np.sum(i))
                proportion.append(temp)
        pshow = []
        for i in proportion:
            pt = "%.2f%%" % (i * 100)
            pshow.append(pt)
        proportion = np.array(proportion).reshape(num_class, num_class)
        pshow = np.array(pshow).reshape(num_class, num_class)
        config = {"font.family": "Times New Roman"}
        rcParams.update(config)
        plt.imshow(proportion, interpolation='nearest',
                   cmap=plt.cm.Blues)  # 按照像素显示出矩阵
        # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
        # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
        plt.title('confusion_matrix')
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, fontsize=12)
        plt.yticks(tick_marks, labels, fontsize=12)
        # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
        # ij配对，遍历矩阵迭代器
        iters = np.reshape([[[i, j] for j in range(num_class)]
                           for i in range(num_class)], (con_matrix.size, 2))
        for i, j in iters:
            if (i == j):
                plt.text(j, i - 0.12, format(con_matrix[i, j]), va='center',
                         ha='center', fontsize=12, color='white', weight=5)  # 显示对应的数字
                plt.text(j, i + 0.12, pshow[i, j], va='center',
                         ha='center', fontsize=12, color='white')
            else:
                # 显示对应的数字
                plt.text(
                    j, i - 0.12, format(con_matrix[i, j]), va='center', ha='center', fontsize=12)
                plt.text(j, i + 0.12, pshow[i, j],
                         va='center', ha='center', fontsize=12)

        plt.ylabel('True label', fontsize=16)
        plt.xlabel('Predict label', fontsize=16)
        plt.tight_layout()
        plt.pause(1)
        plt.show(block=False)
        if not os.path.exists(PIC_SAVED_PATH):
            os.makedirs(PIC_SAVED_PATH)
        save_path = os.path.join(PIC_SAVED_PATH, 'pic.png')
        plt.savefig(save_path)
        print(f'result pic is saved in {PIC_SAVED_PATH}')
        


if __name__ == '__main__':
    # dic_labels = {0: 'W', 1: 'LS', 2: 'SWS', 3: 'REM', 4: 'E'}
    # cm = np.array([(193, 31, 0, 41, 42), (87, 1038, 32, 126, 125),
    #               (17, 337, 862, 1, 2), (17, 70, 0, 638, 54), (1, 2, 3, 4, 5)])
    # matrix_excute = Matrix(None, None)
    # matrix_excute.plot_confusion_matrix(cm, dic_labels)
    y_true = np.array([0]*30 + [1]*240 + [2]*30)
    y_pred = np.array([0]*10 + [1]*10 + [2]*10 + 
                    [0]*40 + [1]*160 + [2]*40 + 
                    [0]*5 + [1]*5 + [2]*20)
    dic_labels = {0:0, 1:1, 2:2}
    matrix_excute = Matrix(y_true=y_true, y_pre=y_pred, mutil=True)
    print(matrix_excute.get_acc())
    print(matrix_excute.get_precision())
    print(matrix_excute.get_recall())
    print(matrix_excute.get_f1())
    matrix_excute.plot_confusion_matrix(dic_labels)

