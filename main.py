#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : main.py
@Time    : 2023/01/14 10:00:39
@Author  : Huang zh
@Contact : jacob.hzh@qq.com
@Version : 0.1
@Desc    : None
'''

import argparse
from process_data import Data_Excuter
from metrics import Matrix
from ml_algorithm.ml_model import ML_EXCUTER
import warnings

warnings.filterwarnings("ignore")

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data path', default=None, type=str)
    parser.add_argument('--model_name', help='model name ex: knn', default=None, type=str)
    parser.add_argument('--model_saved_path', help='the path of model saved', default='./saved/', type=str)
    parser.add_argument('--type_obj', help='need train or test or only predict', default='train', type=str)
    args = parser.parse_args()
    return args

def print_msg(metrix_ex_train, metrix_ex_test, data_ex):
    if metrix_ex_train:
        print('train dataset:')
        print(f"acc: {round(metrix_ex_train.get_acc(), 2)}")
        print(f"presion: {round(metrix_ex_train.get_precision(), 2)}")
        print(f"recall: {round(metrix_ex_train.get_recall(), 2)}")
        print(f"f1: {round(metrix_ex_train.get_f1(), 2)}")
    print('='*20)
    if metrix_ex_test:
        print('test dataset:')
        print(f"acc: {round(metrix_ex_test.get_acc(), 2)}")
        print(f"presion: {round(metrix_ex_test.get_precision(), 2)}")
        print(f"recall: {round(metrix_ex_test.get_recall(), 2)}")
        print(f"f1: {round(metrix_ex_test.get_f1(), 2)}")
        print(metrix_ex_test.plot_confusion_matrix(data_ex.i2l_dic))

def main(args):
    """
    1. 载入数据
    2. 载入模型
    3. 训练模型
    4. 预测结果
    5. 保存模型
    """
    args.data_path = './data/processed_data.csv'
    args.model_name = 'knn'
    args.model_saved_path = './save_model/'
    args.type_obj = 'train'
    if args.type_obj == 'train':
        data_ex = Data_Excuter(args.data_path, split_size=0.3, is_sample=True, split=True)

        model_ex = ML_EXCUTER(args.model_name)
        model_ex.judge_model()

        model_ex.train(data_ex.train_data_x, data_ex.train_data_label)
        
        y_pre_train = model_ex.predict(data_ex.train_data_x)
        y_pre_test = model_ex.predict(data_ex.test_data_x)
        
        mtrix_ex_train = Matrix(data_ex.train_data_label, y_pre_train, mutil=False)
        mtrix_ex_test = Matrix(data_ex.test_data_label, y_pre_test, mutil=False)
        print_msg(mtrix_ex_train, mtrix_ex_test, data_ex)

        model_ex.save_model(args.model_saved_path, args.model_name+'.pkl')
    elif args.type_obj == 'test':
        data_ex = Data_Excuter(args.data_path, split_size=0, is_sample=False, split=False)
        model_ex = ML_EXCUTER(args.model_name)
        model_ex.load_model(args.model_saved_path, args.model_name+'.pkl')
        y_pre_test = model_ex.predict(data_ex.X)
        mtrix_ex_test = Matrix(data_ex.label, y_pre_test, mutil=False)
        print_msg(None, mtrix_ex_test, data_ex)

    else:
        #! todo: 如果是没有标签的，需要直接预测的输出结果文件，例如竞赛提交预测结果，这个看任务要怎么样的数据格式
        # data = 自己设置读取方式
        # model_ex = ML_EXCUTER(args.model_name)
        # model_ex.load_model(args.model_saved_path, args.model_name+'.pkl')
        # y_pre = model_ex.predict(data)
        # 保存的代码，看自己需求是什么
        pass

if __name__ == '__main__':
    args = set_args()
    main(args)