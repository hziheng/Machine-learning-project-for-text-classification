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
from process_data_ml import ML_Data_Excuter
from process_data_dl import DL_Data_Excuter
from metrics import Matrix
from model import Model_Excuter
from config import ML_MODEL_NAME, DL_MODEL_NAME, BATCH_SIZE, SPLIT_SIZE, IS_SAMPLE
from dl_algorithm.dl_config import DlConfig
from trick.set_all_seed import set_seed
import warnings

warnings.filterwarnings("ignore")

# def set_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', help='data path', default='./data/processed_data.csv', type=str)
#     parser.add_argument('--model_name', help='model name ex: knn', default='lg', type=str)
#     parser.add_argument('--model_saved_path', help='the path of model saved', default='./save_model/', type=str)
#     parser.add_argument('--type_obj', help='need train or test or only predict', default='train', type=str)
#     parser.add_argument('--train_data_path', help='train set', default='', type=str)
#     parser.add_argument('--test_data_path', help='test set', default='', type=str)
#     parser.add_argument('--dev_data_path', help='dev set', default='', type=str)
#     args = parser.parse_args()
#     return args


def set_args():
    # 训练代码
    # python --model_name lstm --model_saved_path './save_model/ --type_obj 'train' -- train_data_path './data/dl_data/train.csv' --test_data_path './data/dl_data/test.csv'
    # 测试代码
    # python --model_name lstm --model_saved_path './save_model/ --type_obj 'test' --test_data_path './data/dl_data/test.csv'
    # 预测代码
    # python --model_name lstm --model_saved_path './save_model/ --type_obj 'predict' --dev_data_path './data/dl_data/dev.csv'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data path', default='', type=str)
    parser.add_argument(
        '--model_name', help='model name ex: knn', default='capsules', type=str)
    parser.add_argument(
        '--model_saved_path', help='the path of model saved', default='./save_model/', type=str)
    parser.add_argument(
        '--type_obj', help='need train or test or only predict', default='train', type=str)
    parser.add_argument('--train_data_path',
                        help='train set', default='./data/dl_data/test.csv', type=str)
    parser.add_argument('--test_data_path',
                        help='./data/dl_data/test.csv', default='./data/dl_data/dev.csv', type=str)
    parser.add_argument('--dev_data_path', help='dev set',
                        default='', type=str)
    args = parser.parse_args()
    return args


def print_msg(metrix_ex_train, metrix_ex_test, data_ex):
    if metrix_ex_train:
        print('train dataset:')
        print(f"acc: {round(metrix_ex_train.get_acc(), 4)}")
        print(f"presion: {round(metrix_ex_train.get_precision(), 4)}")
        print(f"recall: {round(metrix_ex_train.get_recall(), 4)}")
        print(f"f1: {round(metrix_ex_train.get_f1(), 4)}")
    print('=' * 20)
    if metrix_ex_test:
        print('test dataset:')
        print(f"acc: {round(metrix_ex_test.get_acc(), 4)}")
        print(f"presion: {round(metrix_ex_test.get_precision(), 4)}")
        print(f"recall: {round(metrix_ex_test.get_recall(), 4)}")
        print(f"f1: {round(metrix_ex_test.get_f1(), 4)}")
        print(metrix_ex_test.plot_confusion_matrix(data_ex.i2l_dic))


def main(args):
    """
    1. 载入数据
    2. 载入模型
    3. 训练模型
    4. 预测结果
    5. 保存模型
    """
    # args.data_path = './data/processed_data.csv'
    # args.model_name = 'knn'
    # args.model_saved_path = './save_model/'
    # args.type_obj = 'train'
    if args.model_name in ML_MODEL_NAME:
        args.model_type = 'ML'
    elif args.model_name in DL_MODEL_NAME:
        args.model_type = 'DL'
    else:
        print('model name error')
        exit(0)

    set_seed(96)

    if args.type_obj == 'train':
        if args.model_type == 'ML':
            data_ex = ML_Data_Excuter(args.data_path, split_size=SPLIT_SIZE, is_sample=IS_SAMPLE,
                                      split=True, train_data_path='', test_data_path='')
            # 初始化模型
            model_ex = Model_Excuter().init(model_name=args.model_name)
        else:
            data_ex = DL_Data_Excuter()
            vocab_size, nums_class = data_ex.process(batch_size=BATCH_SIZE,
                                                     train_data_path=args.train_data_path,
                                                     test_data_path=args.test_data_path,
                                                     dev_data_path=args.dev_data_path)
            dl_config = DlConfig(args.model_name, vocab_size, nums_class, data_ex.vocab)
            # 初始化模型
            model_ex = Model_Excuter().init(dl_config=dl_config)

        model_ex.judge_model()

        # 这里dl和ml的train得用if分开，数据的接口不一样
        if args.model_type == 'ML':
            model_ex.train(data_ex.train_data_x, data_ex.train_data_label)

            y_pre_train = model_ex.predict(data_ex.train_data_x)
            y_pre_test = model_ex.predict(data_ex.test_data_x)

            mtrix_ex_train = Matrix(
                data_ex.train_data_label, y_pre_train, mutil=data_ex.mutil)
            mtrix_ex_test = Matrix(
                data_ex.test_data_label, y_pre_test, mutil=data_ex.mutil)
            print_msg(mtrix_ex_train, mtrix_ex_test, data_ex)

            model_ex.save_model(args.model_saved_path,
                                args.model_name + '.pkl')
        else:
            model_ex.train(data_ex.train_data_loader,
                           data_ex.test_data_loader,
                           data_ex.dev_data_loader,
                           args.model_saved_path,
                           args.model_name + '.pth')

    elif args.type_obj == 'test':
        if args.model_type == 'ML':
            data_ex = ML_Data_Excuter(args.data_path, split_size=0, is_sample=False,
                                      split=False, train_data_path='', test_data_path='')
            model_ex = Model_Excuter().init(model_name=args.model_name)
            model_ex.load_model(args.model_saved_path,
                                args.model_name + '.pkl')
            y_pre_test = model_ex.predict(data_ex.X)
            true_all = data_ex.label

        else:
            data_ex = DL_Data_Excuter()
            vocab_size, nums_class = data_ex.process(
                batch_size=BATCH_SIZE, test_data_path=args.test_data_path)
            dl_config = DlConfig(args.model_name, vocab_size, nums_class, data_ex.vocab)
            # 初始化模型
            model_ex = Model_Excuter().init(dl_config=dl_config)
            model_ex.load_model(args.model_saved_path,
                                args.model_name + '.pth')
            _, _, y_pre_test, true_all = model_ex.evaluate(
                data_ex.test_data_loader)

        mtrix_ex_test = Matrix(true_all, y_pre_test, mutil=data_ex.mutil)
        print_msg(None, mtrix_ex_test, data_ex)

    elif args.type_obj == 'predict':
        if args.model_type == 'ML':
            data_ex = ML_Data_Excuter(args.data_path, split_size=0, is_sample=False,
                                      split=False, train_data_path='', test_data_path='')
            model_ex = Model_Excuter().init(model_name=args.model_name)
            model_ex.load_model(args.model_saved_path,
                                args.model_name + '.pkl')
            y_pre_test = model_ex.predict(data_ex.X)
        else:
            data_ex = DL_Data_Excuter()
            vocab_size, nums_class = data_ex.process(
                batch_size=BATCH_SIZE, dev_data_path=args.dev_data_path)
            dl_config = DlConfig(args.model_name, vocab_size, nums_class, data_ex.vocab)
            # 初始化模型
            model_ex = Model_Excuter().init(dl_config=dl_config)
            model_ex.load_model(args.model_saved_path,
                                args.model_name + '.pth')
            y_pre_test = model_ex.predict(data_ex.dev_data_loader)
            # data_ex.i2l_dic可以将y_pre_test中的数字转成文字标签，按需使用
        #! 如何保存数据，按需求填写
    else:
        print('please input train, test or predict in type_obj of params!')
        exit(0)


if __name__ == '__main__':
    args = set_args()
    main(args)