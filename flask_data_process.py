# -*- coding: utf-8 -*-
# __author : Bossun_Chen
# __time : 2021/12/6 上午 09:07

import os
import config
import logging
import numpy as np

class GetData:

    def __init__(self,infos):
        self.infos = infos
        self.config = config
        # self.data_dir = config.data_dir

    def all_list(self,text,step=511):
        word_lis = list()
        for i, list_name in enumerate(text):
            if i % step == 0:
                word_lis.append(list())
            word_lis[-1].append(list_name)
        return word_lis

    def preprocess(self, mode="predict1"):
        """
        params:
            words：将json文件每一行中的文本分离出来，存储为words列表
            labels：标记文本对应的标签，存储为labels
        examples:
            words示例：['一', '例','马', '来', '西', '亚', '输', '入']
            labels示例：['O', 'O', 'B-domestic', 'I-domestic', 'I-domestic', 'I-domestic', 'O', 'O']
        """

        word_lis = []
        label_lis = []

        texts = list(self.infos.strip())
        text = [text.replace('\xa0', '-').replace('\t', ',').replace('\ue0e7', '-').replace('\u3000', '-').replace('\n', '。') for text in
                texts]
        tag_list = ['O' for i in range(len(text))]

        word_lis.extend(self.all_list(text))
        label_lis.extend(self.all_list(tag_list))

        return word_lis, label_lis
