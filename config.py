# -*- coding: utf-8 -*-
# __author : Bossun_Chen
# __time : 2021/11/22 上午 09:39

import os
import torch

data_dir = os.getcwd() + '/data/clue/'
train_dir = data_dir + 'train.npz'
files = ['train']
# predict_dir = data_dir + 'predict.npz'
# files = ['train','predict']
bert_model = 'pretrained_bert_models/bert-base-chinese/'
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
model_dir = os.getcwd() + '/experiments/clue/'
model_dir1 = os.getcwd() + '/experiments/clue1/'
log_dir = model_dir + 'train.log'
log_dir = model_dir1 + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 32
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = ''

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

labels = ['name', 'workplace', 'occupation', 'native', 'abroad',
          'domestic', 'port', 'insulate', 'hospital', 'date',
          'person', 'location', 'mask', 'airline', 'license',
          'symptom', 'test', 'IG', 'CT', 'blood']

label2id = {
    "O": 0,
    "B-name": 1,
    "B-workplace": 2,
    "B-occupation": 3,
    'B-native': 4,
    'B-abroad': 5,
    'B-domestic': 6,
    'B-port': 7,
    'B-insulate': 8,
    'B-hospital': 9,
    'B-date': 10,
    'B-person': 11,
    'B-location': 12,
    'B-mask': 13,
    'B-airline': 14,
    'B-license': 15,
    'B-symptom': 16,
    'B-test': 17,
    'B-IG': 18,
    'B-CT': 19,
    'B-blood': 20,
    "I-name": 21,
    "I-workplace": 22,
    "I-occupation": 23,
    'I-native': 24,
    'I-abroad': 25,
    'I-domestic': 26,
    'I-port': 27,
    'I-insulate': 28,
    'I-hospital': 29,
    'I-date': 30,
    'I-person': 31,
    'I-location': 32,
    'I-mask': 33,
    'I-airline': 34,
    'I-license': 35,
    'I-symptom': 36,
    'I-test': 37,
    'I-IG': 38,
    'I-CT': 39,
    'I-blood': 40
}


id2label = {_id: _label for _label, _id in list(label2id.items())}

# labels = ['name', 'workplace', 'occupation', 'native', 'abroad',
#           'domestic', 'port', 'insulate', 'hospital', 'date',
#           'person', 'location', 'mask', 'airline', 'license',
#           'symptom', 'test', 'IG', 'CT', 'blood', 'vaccine']

# label2id = {
#     "O": 0,
#     "B-name": 1,
#     "B-workplace": 2,
#     "B-occupation": 3,
#     'B-native': 4,
#     'B-abroad': 5,
#     'B-domestic': 6,
#     'B-port': 7,
#     'B-insulate': 8,
#     'B-hospital': 9,
#     'B-date': 10,
#     'B-person': 11,
#     'B-location': 12,
#     'B-mask': 13,
#     'B-airline': 14,
#     'B-license': 15,
#     'B-symptom': 16,
#     'B-test': 17,
#     'B-IG': 18,
#     'B-CT': 19,
#     'B-blood': 20,
#     'B-vaccine':21,
#     "I-name": 22,
#     "I-workplace": 23,
#     "I-occupation": 24,
#     'I-native': 25,
#     'I-abroad': 26,
#     'I-domestic': 27,
#     'I-port': 28,
#     'I-insulate': 29,
#     'I-hospital': 30,
#     'I-date': 31,
#     'I-person': 32,
#     'I-location': 33,
#     'I-mask': 34,
#     'I-airline': 35,
#     'I-license': 36,
#     'I-symptom': 37,
#     'I-test': 38,
#     'I-IG': 39,
#     'I-CT': 40,
#     'I-blood': 41,
#     'I-vaccine': 42
# }