# -*- coding: utf-8 -*-
# __author : Bossun_Chen
# __time : 2021/11/22 上午 11:59

import utils
import config
import logging
import numpy as np
from data_process import Processor
from data_loader import NERDataset
from model import BertNER
from train import train, evaluate
from flask_data_process import GetData

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

def dev_split(dataset_dir):
    """split dev set"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev

def load_dev(mode):
    if mode == 'train':
        # 分离出验证集
        word_train, word_dev, label_train, label_dev = dev_split(config.train_dir)
    elif mode == 'test':
        train_data = np.load(config.train_dir, allow_pickle=True)
        dev_data = np.load(config.test_dir, allow_pickle=True)
        word_train = train_data["words"]
        label_train = train_data["labels"]
        word_dev = dev_data["words"]
        label_dev = dev_data["labels"]
    else:
        word_train = None
        label_train = None
        word_dev = None
        label_dev = None
    return word_train, word_dev, label_train, label_dev

def test():
    data = np.load(config.test_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    test_dataset = NERDataset(word_test, label_test, config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    # Prepare model
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        return
    val_metrics = evaluate(test_loader, model, mode='test')
    val_f1 = val_metrics['f1']
    logging.info("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_f1))
    val_f1_labels = val_metrics['f1_labels']
    for label in config.labels:
        logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))

def predict(content):
    data = GetData(content)
    text = data.preprocess()
    word_pre = text[0]
    label_pre = text[1]
    pre_dataset = NERDataset(word_pre,label_pre,config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    pre_loader = DataLoader(pre_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=pre_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")

    # Prepare model

    # name/license
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to predict !--------")
        return

    # date/location
    if config.model_dir1 is not None:
        model1 = BertNER.from_pretrained(config.model_dir1)
        model1.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir1))
    else:
        logging.info("--------No model to predict !--------")
        return

    pre_metrics = evaluate(pre_loader, model, mode='pre')
    pre_metrics1 = evaluate(pre_loader, model1, mode='pre')
    pre_label = pre_metrics['pre_labels']
    pre_label1 = pre_metrics1['pre_labels']
    return pre_label, pre_label1


def run():
    """train the model"""
    # set the logger
    utils.set_logger(config.log_dir)
    logging.info("device: {}".format(config.device))
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.process()
    logging.info("--------Process Done!--------")
    # 分离出验证集
    word_train, word_dev, label_train, label_dev = load_dev('train')
    # build dataset
    train_dataset = NERDataset(word_train, label_train, config)
    dev_dataset = NERDataset(word_dev, label_dev, config)
    logging.info("--------Dataset Build!--------")
    # get dataset size
    train_size = len(train_dataset)
    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    logging.info("--------Get Dataloader!--------")
    # for x, y, z in train_loader:
    #     print("batch_data.shape: ", x.shape)
    #     print("batch_label_starts.shape: ", y.shape)
    #     print("batch_labels.shape: ", z.shape)
    # Prepare model
    device = config.device
    model = BertNER.from_pretrained(config.roberta_model, num_labels=len(config.label2id))
    model.to(device)
    # Prepare optimizer
    if config.full_fine_tuning:
        # model.named_parameters(): [bert, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
        ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)

    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)


    # 如果训练集不够，用这个计算分数.
    val_metrics = evaluate(dev_loader, model, mode='test')
    val_f1 = val_metrics['f1']
    logging.info("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_f1))
    val_f1_labels = val_metrics['f1_labels']
    for label in config.labels:
        logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))

if __name__ == '__main__':
    run()
    # test()
    # predict = predcit()
    # print("native:", predict['native'])
    # print("abroad:", predict['abroad'])
    # print("hospital:", predict['hospital'])
    # print("port:", predict['port'])
    # print("insulate:", predict['insulate'])
    # print("domestic:", predict['domestic'])


    # labels = ['name', 'workplace', 'occupation', 'native', 'abroad',
    #           'domestic', 'port', 'insulate', 'hospital', 'date',
    #           'person', 'location', 'mask', 'airline', 'license',
    #           'symptom', 'test', 'IG', 'CT', 'blood']
