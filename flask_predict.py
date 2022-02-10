# -*- coding: utf-8 -*-
# __author : Bossun_Chen
# __time : 2021/12/9 下午 06:08

import config
import logging

from model import BertNER
from data_loader import NERDataset
from train import train, evaluate
from flask_data_process import GetData
from torch.utils.data import DataLoader

class PredictModel(object):

    def __init__(self):

        # Prepare model

        # name/license
        if config.model_dir is not None:
            self.model1 = BertNER.from_pretrained(config.model_dir)
            self.model1.to(config.device)
            logging.info("--------Load model from {}--------".format(config.model_dir))
        else:
            logging.info("--------No model to predict !--------")
            return

        # date/location
        if config.model_dir1 is not None:
            self.model2 = BertNER.from_pretrained(config.model_dir1)
            self.model2.to(config.device)
            logging.info("--------Load model from {}--------".format(config.model_dir1))
        else:
            logging.info("--------No model to predict !--------")
            return

    def predict(self, content):
        data = GetData(content)
        text = data.preprocess()
        word_pre = text[0]
        label_pre = text[1]
        pre_dataset = NERDataset(word_pre, label_pre, config)
        logging.info("--------Dataset Build!--------")
        # build data_loader
        pre_loader = DataLoader(pre_dataset, batch_size=config.batch_size,
                                shuffle=False, collate_fn=pre_dataset.collate_fn)
        logging.info("--------Get Data-loader!--------")


        pre_metrics = evaluate(pre_loader, self.model1, mode='pre')
        pre_metrics1 = evaluate(pre_loader, self.model2, mode='pre')
        pre_label = pre_metrics['pre_labels']
        pre_label1 = pre_metrics1['pre_labels']
        return pre_label, pre_label1

    def align(self, datas, content):
        ner_align = []
        for i, data in enumerate(datas):
            data_temp = data
            for j in content:
                if j == '':
                    continue
                if j in data_temp:
                    ner_align.append(j)
                    data_temp = data_temp.replace(j, '')
            if len(ner_align) == 0:
                ner_align.append('')
        return ner_align

    def time_slice(self, dates):
        times  = {
            'startTravelDate': "",      # 开始日期(yyyyMMdd)
            'endTravelDate': "",        # 结束日期(yyyyMMdd)
            'startTravelHour': "",      # 开始时间(hh24:mm)
            'endTravelHour': "",        # 结束时间(hh24:mm)
            'travelDuration': "",       # 停留时长(单位:分钟)
            'startTravelDateStr': "",   # 开始日期(文本)
            'endTravelDateStr': "",     # 结束日期(文本)
            'startTravelHourStr': "",   # 开始时间(文本)
            'endTravelHourStr': "",     # 结束时间(文本)
            'travelDurationStr': "",    # 停留时长(文本)
        }
        return None