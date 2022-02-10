# -*- coding: utf-8 -*-
# __author : Bossun_Chen
# __time : 2021/11/21 下午 04:36

import os
import json
import config
import logging
import numpy as np


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config


    def process(self):
        """
        process train and test data
        """
        for file_name in self.config.files:
            self.preprocess(file_name)

    def all_list(self,text,step=511):
        word_lis = list()
        for i, list_name in enumerate(text):
            if i % step == 0:
                word_lis.append(list())
            word_lis[-1].append(list_name)
        return word_lis

    def creat_BIO(self,text,labels):
        tag_list = ['O' for i in range(len(text))]
        for start_index, end_index, key in labels:
            tag_list[start_index] = 'B-' + str(key)
            k = start_index + 1
            while k < end_index:
                tag_list[k] = 'I-' + str(key)
                k +=1
        return tag_list

    def preprocess(self, mode):
        """
        params:
            words：将json文件每一行中的文本分离出来，存储为words列表
            labels：标记文本对应的标签，存储为labels
        examples:
            words示例：['一', '例','马', '来', '西', '亚', '输', '入']
            labels示例：['O', 'O', 'B-domestic', 'I-domestic', 'I-domestic', 'I-domestic', 'O', 'O']
        """
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return

        word_lis = []
        label_lis = []
        with open(input_dir, 'r', encoding='utf-8') as f:
            json_line = json.load(f)

        for i in range(len(json_line)):
            # labels = json_line[i]['labels']
            label_entities = json_line[i].get('labels', None)
            text = json_line[i]['text']
            texts = list(text.strip())

            # 数据清洗
            text = [text.replace('\xa0', '-').replace('\t', ',').replace('\ue0e7', '-').replace('\u3000', '-').replace('\n', '。').replace(' ', '-') for text in texts]

            if label_entities is not None:
                tag_list = self.creat_BIO(text, label_entities)
            else:
                tag_list = ['O' for i in range(len(text))]
            word_lis.extend(self.all_list(text))
            label_lis.extend(self.all_list(tag_list))


        for i in range(len(word_lis)):
            print("word_list{%s}:" % i, word_lis[i])
        # print("word_list:", len(word_lis))
        # print("word_lis:", word_lis[5])
        # print("word_lis:", word_lis[1])
        # print("word_lis:", len(word_lis[1682]))
        # print("word_lis:", len(word_lis))

        # return np.asarray(word_lis),np.asarray(label_lis)

        # # 保存成二进制文件
        # np.savez_compressed(output_dir, words=word_lis, labels=label_lis,dtype=object)
        # logging.info("--------{} data process DONE!--------".format(mode))


if __name__ == '__main__':

    processer = Processor(config)
    processer.process()

    # def consist(text,step=511):
    #     word_lis = list()
    #     for i in range(len(text)):
    #         if i % step==0:
    #             word_lis.append(list())
    #         word_lis[-1].append(text[i])
    #     return word_lis
    #
    # print(consist('fdjsqyhdehdvdhfc',6))
    #
    # def consist(text,step=511):
    #     word_lis = list()
    #     for i,char in enumerate(text):
    #         if i % step==0:
    #             word_lis.append(list())
    #         word_lis[-1].append(char)
    #     return word_lis
    #
    # print(consist('fdjsqyhdehdvdhfc',6))
