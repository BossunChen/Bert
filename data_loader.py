# -*- coding: utf-8 -*-
# __author : Bossun_Chen
# __time : 2021/11/22 上午 11:57

import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, words, labels, config, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
        self.label2id = config.label2id
        self.id2label = {_id: _label for _label, _id in list(config.label2id.items())}
        self.dataset = self.preprocess(words, labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = config.device

    def preprocess(self, origin_sentences, origin_labels):
        """
        Maps tokens and tags to their indices and stores them in the dict data.
        examples:
            word:['[CLS]', '关', '于', '1', '0', '月', '2', '8', '日', '广']
            sentence:([101, 1068, 754, 122, 121, 3299, 123, 129, 3189, 2408],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
            label:[3, 13, 13, 13, 0, 0, 0, 0, 0]
        """
        data = []
        sentences = []
        labels = []

        for line in origin_sentences:
            # replace each token by its index
            # we can not use encode_plus because our sentences are aligned to labels in list type
            words = []
            word_lens = []
            for token in line:
                words.append(self.tokenizer.tokenize(token))
                word_lens.append(len(token))
            print("words:", words)
            print("words_len:", len(words))

            # 变成单个字的列表，开头加上[CLS]
            words = ['[CLS]'] + [item for token in words for item in token]
            print("words_cls:", words)
            print("words_cls_len:", len(words))
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs))

        for tag in origin_labels:
            label_id = [self.label2id.get(t) for t in tag]
            labels.append(label_id)

        for sentence, label, origin_sentence in zip(sentences, labels, origin_sentences):
            data.append((sentence, label, origin_sentence))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        origin_content = self.dataset[idx][2]
        return [word, label, origin_content]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        process batch data, including:
            1. padding: 将每个batch的data padding到同一长度（batch中最长的data长度）
            2. aligning: 找到每个sentence sequence里面有label项，文本与label对齐
            3. tensor：转化为tensor
            4. batch 是 batch_size 目前设置为1
        """
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        origin_contents = [x[2] for x in batch]

        # batch length
        batch_len = len(sentences)

        # compute length of longest sentence in batch
        max_origin_len = max([len(s) for s in origin_contents]) # 78
        max_len = max([len(s[0]) for s in sentences])
        max_label_len = 0

        # padding data 初始化
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len)) # 79
        batch_label_starts = []

        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])                 # 79
            batch_data[j][:cur_len] = sentences[j][0]      # batch_data 101, 1068, 754, 122, 121, 3299, 123
            # 找到有标签的数据的index（[CLS]不算）
            label_start_idx = sentences[j][-1]             # len 78
            label_starts = np.zeros(max_len)               # len 79
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)        # batch_label_starts 0,1,1,1,1,1,1,,1,1,1,1
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len)) # 78
        for j in range(batch_len):
            cur_tags_len = len(labels[j])                  # 78
            batch_labels[j][:cur_tags_len] = labels[j]     # 00000000000000
        
        # original content
        batch_original_contents = np.ones((batch_len, max_origin_len),dtype=str)
        for j in range(batch_len):
            cur_original_len = len(origin_contents[j])
            batch_original_contents[j][:cur_original_len] = origin_contents[j]

        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)

        return [batch_data, batch_label_starts, batch_labels, batch_original_contents]
