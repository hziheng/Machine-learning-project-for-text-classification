# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Author       : Huang zh
 Email        : jacob.hzh@qq.com
 Date         : 2023-03-19 10:48:29
 LastEditTime : 2023-03-19 10:59:57
 FilePath     : \\Machine-learning-project-for-text-classification\\word2vec_train.py
 Description  : 
'''


import os
import pickle
import argparse
from gensim.models import word2vec, keyedvectors
from gensim.models.callbacks import CallbackAny2Vec



def pickle_read(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data



# 定义回调函数
class callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1

def input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='input dir name', default='./result_pickle_1')
    parser.add_argument('-o', '--outputfile', help='output file name', default='./result_pickle_1')
    args = parser.parse_args()
    print(args)
    return args


def word2vec_train(stences, only_vec=False):
    if only_vec:
        if os.path.exists("w2v_vec_300.bin.gz"):
            model = keyedvectors.load_word2vec_format("w2v_vec_300.bin.gz", binary=True)
            return model
        else:
            vec_path = 'w2v_vec_300.bin.gz'
            # save model, word_vectors
            model = word2vec.Word2Vec(sentences=stences, min_count=5, vector_size=300, epochs=100, callbacks=[callback()],compute_loss=True, workers=16)
            model.wv.save_word2vec_format(vec_path, binary=True)
            return model.wv

    else:
        if os.path.exists("w2v_model.bin"):
            model = word2vec.Word2Vec.load("w2v_model.bin")
        else:
            model = word2vec.Word2Vec(sentences=stences, min_count=5, vector_size=300,  epochs=100, callbacks=[callback()],compute_loss=True, workers=16)
            model.save("w2v_model.bin")
            model.wv.save_word2vec_format('./embed.txt')
        return model.wv



def main(args):
    # all_data_tokens = word_token(args.input_dir)
    with open('./d.pkl', 'rb') as f:
        all_data_tokens = pickle.load(f)
    print('train begin')
    model = word2vec_train(all_data_tokens, only_vec=False)
    print('train over')
    print(model.get_vector('我'))

def test_model():
    # model = word2vec.Word2Vec.load("w2v_model.bin")
    # print(model.wv.get_vector('00'))
    model = keyedvectors.load_word2vec_format('w2v_vec_300.bin.gz', binary=True)
    print(model.get_vector('我'))

if __name__ == '__main__':
    args = input()
    main(args)
    # test_model()
