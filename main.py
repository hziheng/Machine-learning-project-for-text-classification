# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Author       : Huang zh
 Email        : jacob.hzh@qq.com
 Date         : 2023-03-09 19:27:58
 LastEditTime : 2023-03-23 15:16:11
 FilePath     : \\codes\\main.py
 Description  : 
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import transformers
from process_data_ml import ML_Data_Excuter
from process_data_dl import DL_Data_Excuter
from process_data_pretrain import PRE_Data_Excuter
from metrics import Matrix
from model import Model_Excuter
from config import ML_MODEL_NAME, DL_MODEL_NAME, PRE_MODEL_NAME, BATCH_SIZE, SPLIT_SIZE, IS_SAMPLE
from dl_algorithm.dl_config import DlConfig
from trick.set_all_seed import set_seed
import warnings


warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# def set_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', help='data path', default='', type=str)
#     parser.add_argument(
#         '--model_name', help='model name ex: knn', default='lg', type=str)
#     parser.add_argument(
#         '--model_saved_path', help='the path of model saved', default='./save_model/', type=str)
#     parser.add_argument(
#         '--type_obj', help='need train or test or only predict', default='test', type=str)
#     parser.add_argument('--train_data_path',
#                         help='train set', default='', type=str)
#     parser.add_argument('--test_data_path', help='test set',
#                         default='./data/processed_data.csv', type=str)
#     parser.add_argument('--dev_data_path', help='dev set',
#                         default='', type=str)
#     args = parser.parse_args()
#     return args


def set_args():
    # 训练代码
    # python main.py --model_name transformer --model_saved_path ./save_model/ --type_obj train --train_data_path ./data/dl_data/test.csv --test_data_path ./data/dl_data/dev.csv
    # 测试代码
    # python main.py --model_name lstm --model_saved_path ./save_model/ --type_obj test --test_data_path ./data/dl_data/test.csv
    # 预测代码
    # python main.py --model_name lstm --model_saved_path './save_model/ --type_obj predict --dev_data_path ./data/dl_data/dev.csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data path', default='', type=str)
    parser.add_argument(
        '--model_name', help='model name ex: knn', default='transformer', type=str)
    parser.add_argument(
        '--model_saved_path', help='the path of model saved', default='./save_model/transformer', type=str)
    parser.add_argument(
        '--type_obj', help='need train or test or only predict', default='train', type=str)
    parser.add_argument('--train_data_path',
                        help='train set', default='./data/dl_data/test.csv', type=str)
    parser.add_argument('--test_data_path',
                        help='./data/dl_data/test.csv', default='./data/dl_data/dev.csv', type=str)
    parser.add_argument('--dev_data_path', help='dev set',
                        default='', type=str)
    parser.add_argument('--pretrain_file_path', help='# 预训练模型的文件地址（模型在transformers官网下载）',
                        default='./pretrain_model/roberta_wwm/', type=str)
    args = parser.parse_args()
    return args


def print_msg(metrix_ex_train, metrix_ex_test, data_ex, pic_name='pic'):
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
        print(metrix_ex_test.plot_confusion_matrix(data_ex.i2l_dic, pic_name))


def create_me_de(args, split_size=SPLIT_SIZE, is_sample=IS_SAMPLE, split=True, batch_size=BATCH_SIZE, train_data_path='', test_data_path='', need_predict=False):
    if args.model_type == 'ML':
        data_ex = ML_Data_Excuter(args.data_path, split_size=split_size, is_sample=is_sample,
                                  split=split, train_data_path=train_data_path, test_data_path=test_data_path)
        # 初始化模型
        model_ex = Model_Excuter().init(model_name=args.model_name)
        if need_predict and args.type_obj == 'test':
            model_ex.load_model(args.model_saved_path,
                                args.model_name + '.pkl')
            y_pre_test = model_ex.predict(data_ex.X)
            true_all = data_ex.label
            return data_ex, model_ex, true_all, y_pre_test
        elif need_predict and args.type_obj == 'predict':
            model_ex.load_model(args.model_saved_path,
                                args.model_name + '.pkl')
            y_pre_test = model_ex.predict(data_ex.X)
            return data_ex, model_ex, y_pre_test
    elif args.model_type == 'DL':
        data_ex = DL_Data_Excuter()
        vocab_size, nums_class = data_ex.process(batch_size=batch_size,
                                                 train_data_path=args.train_data_path,
                                                 test_data_path=args.test_data_path,
                                                 dev_data_path=args.dev_data_path)
        dl_config = DlConfig(args.model_name, vocab_size,
                             nums_class, data_ex.vocab)
        # 初始化模型
        model_ex = Model_Excuter().init(dl_config=dl_config)
        if need_predict and args.type_obj == 'test':
            model_ex.load_model(args.model_saved_path,
                                args.model_name + '.pth')
            _, _, y_pre_test, true_all = model_ex.evaluate(
                data_ex.test_data_loader)
            return data_ex, model_ex, true_all, y_pre_test
        elif need_predict and args.type_obj == 'predict':
            model_ex.load_model(args.model_saved_path,
                                args.model_name + '.pth')
            y_pre_test = model_ex.predict(data_ex.dev_data_loader)
            return data_ex, model_ex, y_pre_test
    else:
        data_ex = PRE_Data_Excuter(args.model_name)
        nums_class = data_ex.process(batch_size=batch_size,
                                     train_data_path=args.train_data_path,
                                     test_data_path=args.test_data_path,
                                     dev_data_path=args.dev_data_path,
                                     pretrain_file_path=args.pretrain_file_path
                                     )
        dl_config = DlConfig(args.model_name, 0, nums_class, '', 'random')
        # 初始化模型
        model_ex = Model_Excuter().init(dl_config=dl_config)
        if need_predict and args.type_obj == 'test':
            model_ex.load_model(args.model_saved_path)
            _, _, y_pre_test, true_all = model_ex.evaluate(
                data_ex.test_data_loader)
            return data_ex, model_ex, true_all, y_pre_test
        elif need_predict and args.type_obj == 'predict':
            model_ex.load_model(args.model_saved_path)
            y_pre_test = model_ex.predict(data_ex.dev_data_loader)
            return data_ex, model_ex, y_pre_test

    return data_ex, model_ex


def main(args):
    """
    1. 载入数据
    2. 载入模型
    3. 训练模型
    4. 预测结果
    5. 保存模型
    """
    if args.model_name in ML_MODEL_NAME:
        args.model_type = 'ML'
    elif args.model_name in DL_MODEL_NAME:
        args.model_type = 'DL'
    elif args.model_name in PRE_MODEL_NAME:
        args.model_type = 'PRE'
    else:
        print('model name error')
        exit(0)

    set_seed(96)

    if args.type_obj == 'train':
        data_ex, model_ex = create_me_de(args)

        model_ex.judge_model(args.pretrain_file_path)

        # 这里dl和ml的train得用if分开，数据的接口不一样
        if args.model_type == 'ML':
            model_ex.train(data_ex.train_data_x, data_ex.train_data_label)

            y_pre_train = model_ex.predict(data_ex.train_data_x)
            y_pre_test = model_ex.predict(data_ex.test_data_x)

            mtrix_ex_train = Matrix(
                data_ex.train_data_label, y_pre_train, multi=data_ex.multi)
            mtrix_ex_test = Matrix(
                data_ex.test_data_label, y_pre_test, multi=data_ex.multi)
            print_msg(mtrix_ex_train, mtrix_ex_test, data_ex, 'train_pic')

            model_ex.save_model(args.model_saved_path,
                                args.model_name + '.pkl')
        elif args.model_type == 'DL':
            model_ex.train(data_ex.train_data_loader,
                           data_ex.test_data_loader,
                           data_ex.dev_data_loader,
                           args.model_saved_path,
                           args.model_name + '.pth')
        else:
            model_ex.dlconfig.pretrain_file_path = args.pretrain_file_path
            model_ex.train(data_ex.train_data_loader,
                           data_ex.test_data_loader,
                           data_ex.dev_data_loader,
                           args.model_saved_path
                           )

    elif args.type_obj == 'test':
        args.data_path = args.test_data_path
        args.train_data_path, args.dev_data_path = '', ''
        data_ex, model_ex, true_all, y_pre_test = create_me_de(
            args, split_size=0, is_sample=False, split=False, need_predict=True)
        mtrix_ex_test = Matrix(true_all, y_pre_test, multi=data_ex.multi)
        print_msg(None, mtrix_ex_test, data_ex, 'test_pic')

    elif args.type_obj == 'predict':
        args.data_path = args.dev_data_path
        args.train_data_path, args.test_data_path = '', ''
        data_ex, model_ex, y_pre_test = create_me_de(
            args, split_size=0, is_sample=False, split=False, need_predict=True)
        # data_ex.i2l_dic可以将y_pre_test中的数字转成文字标签，按需使用
        #! 如何保存数据，按需求填写
    else:
        print('please input train, test or predict in type_obj of params!')
        exit(0)


if __name__ == '__main__':
    args = set_args()
    main(args)
