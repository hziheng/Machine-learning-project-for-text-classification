# !usr/bin/env python
# -*- coding:utf-8 -*-

'''
 Author       : Huang zh
 Email        : jacob.hzh@qq.com
 Date         : 2023-03-13 15:09:48
 LastEditTime : 2023-03-21 19:51:13
 FilePath     : \\codes\\process_data_pretrain.py
 Description  : data process for pretrain method
'''


from process_data_dl import DataSetProcess
from trick.dynamic_padding import collater
from torch.utils.data import Dataset, DataLoader
from config import MAX_SEQ_LEN
from transformers import AutoTokenizer


class DataSetProcess_pre(DataSetProcess):
    def trans_data(self, data_path, label_dic):
        contents = []
        datas, labels = self.load_data(data_path)
        if not labels:
            labels = [-1] * len(datas)
        for d, l in zip(datas, labels):
            if not d.strip():
                continue
            if l != -1:
                contents.append(([d], int(label_dic.get(l))))
            else:
                contents.append(([d],))
        return contents


class PREDataset(Dataset):
    def __init__(self, contents, tokenizer, max_seq_len):
        self. tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data, self.label = self.get_data_label(contents)
        self.len = len(self.data)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        #! todo 增量训练的数据集构造
        # 预测返回的label和上面不一样，因为预测是和任务有关，不是做pretrain，因此要用数据集的标签，上面要用自己的数据增量训练，所以要自己预测mask的地方
        tokenize_result = self.tokenizer.encode_plus(self.data[index], max_length=self.max_seq_len)
        if self.label:
            return {
                'input_ids': tokenize_result["input_ids"],
                'attention_mask': tokenize_result["attention_mask"],
                'token_type_ids': tokenize_result["token_type_ids"],
                'label': self.label[index]

            }
        return {
            'input_ids': tokenize_result["input_ids"],
            'attention_mask': tokenize_result["attention_mask"],
            'token_type_ids': tokenize_result["token_type_ids"],
        }

    def get_data_label(self, contents):
        """contents: [([data], label?), ()]
        """
        data = []
        label = []
        for i in contents:
            data.append(i[0][0])
            if len(i) == 2:
                label.append(i[1])
        return data, label


class PRE_Data_Excuter:
    def __init__(self, model_type):
        self.model_type = model_type
        
    def process(self,batch_size, train_data_path='', test_data_path='', dev_data_path='', pretrain_file_path=''):
        self.pretrain_file_path = pretrain_file_path
        #* 分词器的设置，不同模型不一样的分词器
        # if self.model_type in ['mac_bert','bert', 'bert_wwm', 'nezha_wwm']:
        #     from transformers import BertTokenizer
        #     tokenizer = BertTokenizer.from_pretrained(self.pretrain_file_path)
        # #// 其他分词器，先不用Autotokenizer这个类
        # else:
        #     print('tokenizer is null, please check model_name')
        #     exit()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrain_file_path)
        p = DataSetProcess_pre(train_data_path, test_data_path, dev_data_path)
        self.label_dic, self.i2l_dic = p.build_label2id(save=True)
        if len(self.label_dic) > 2:
            self.multi = True
        else:
            self.multi = False
        collater_fn = collater(pad_index=0, for_pretrain=True)
        self.train_data_loader = ''
        self.test_data_loader = ''
        self.dev_data_loader = ''
        if train_data_path:
            content = p.trans_data(train_data_path, self.label_dic)
            data_set = PREDataset(content,tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
            self.train_data_loader = DataLoader(
                data_set, batch_size=batch_size, shuffle=True, collate_fn=collater_fn)
        if test_data_path:
            content = p.trans_data(test_data_path, self.label_dic)
            data_set = PREDataset(content,tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
            self.test_data_loader = DataLoader(
                data_set, batch_size=batch_size, shuffle=False, collate_fn=collater_fn)
        if dev_data_path:
            content = p.trans_data(dev_data_path, self.label_dic)
            data_set = PREDataset(content,tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
            self.dev_data_loader = DataLoader(
                data_set, batch_size=batch_size, shuffle=False, collate_fn=collater_fn)
        return len(self.label_dic)

