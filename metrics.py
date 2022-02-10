# -*- coding: utf-8 -*-
# __author : Bossun_Chen
# __time : 2021/11/23 下午 03:56

import os
import config
import logging


def get_entities(seq):
    """
    Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'S':
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, mode='dev'):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    if mode == 'dev':
        return score
    else:
        f_score = {}
        for label in config.labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            f_score[label] = score_label
        return f_score, score


def bad_case(y_true, y_pred, data):
    if not os.path.exists(config.case_dir):
        os.system(r"touch {}".format(config.case_dir))  # 调用系统命令行来创建文件
    output = open(config.case_dir, 'w',encoding='utf-8')
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue
        else:
            output.write("bad case " + str(idx) + ": \n")
            output.write("sentence: " + str(data[idx]) + "\n")
            output.write("golden label: " + str(t) + "\n")
            output.write("model pred: " + str(p) + "\n")

    logging.info("--------Bad Cases reserved !--------")

def make_join(y_pred,data):
    sentences = []
    for idx, p in enumerate(y_pred):
        sentence = []
        sentence.append(''.join(data[idx]))
        sentences.append(sentence)
    return sentences


def pre_label(y_pred,data):
    map_dict = dict()
    for label in config.labels:
        map_dict[label] = []

    sentences = make_join(y_pred,data)

    entity_name = ""
    flag = []
    for i, _ in enumerate(sentences):
        for char, tag in zip(sentences[i][0], y_pred[i]):
            if tag[0] == "B":
                if entity_name != "":
                    x = dict((a, flag.count(a)) for a in flag)
                    y = [k for k, v in x.items() if max(x.values()) == v]
                    for label in config.labels:
                        if y[0] in label:
                            map_dict[label].append(entity_name)
                    flag.clear()
                    entity_name = ""
                entity_name += char
                flag.append(tag[2:])
            elif tag[0] == "I":
                entity_name += char
                flag.append(tag[2:])
            else:
                if entity_name != "":
                    x = dict((a, flag.count(a)) for a in flag)
                    y = [k for k, v in x.items() if max(x.values()) == v]
                    for label in config.labels:
                        if y[0] in label:
                            map_dict[label].append(entity_name)
                    flag.clear()
                flag.clear()
                entity_name = ""
    return map_dict

if __name__ == "__main__":
    y_t = [['O', 'O', 'O', 'B-address', 'I-address', 'I-address', 'O'], ['B-name', 'I-name', 'O']]
    y_p = [['O', 'O', 'B-address', 'I-address', 'I-address', 'I-address', 'O'], ['B-name', 'I-name', 'O']]
    sents = [['十', '一', '月', '中', '山', '路', '电'], ['周', '静', '说']]
    bad_case(y_t, y_p, sents)
    # word = []
    # print(len(sents))
    # for j in range(len(sents)):
    #     word_s = []
    #     sent = sents[j]
    #     for i in range(len(sent)):
    #         if y_p[j][i] != 'O':
    #             word_s.append(sent[i])
    #     word.append(word_s)
    # print(word)

    sentence= ['生', '活', '轨', '迹', '比', '较', '固', '定', '，', '无', '业', '在', '家', '，', '偶', '尔', '外', '出', '超', '市', '购',
               '买', '日', '用', '品', '，', '自', '述', '外', '出', '史', '均', '有', '佩', '戴', '一', '次', '性', '口', '罩', '；', '病',
               '例', 'f', 'u', 's', 'h', 'a', 'n', 'j', 'a', 'b', 'd', 'u', 'l', 'h', 'a', 'd', 'i', '2', '0', '1', '9',
               '年', '1', '月', '从', '中', '国', '深', '圳', '返', '回', '土', '耳', '其', '伊', '斯', '坦', '布', '尔', '再', '无', '外',
               '出', '，', '在', '当', '地', '从', '事', '手', '机', '销', '售', '业', '务', '。', '2', '0', '2', '0', '年', '日', '常',
               '生', '活', '轨', '迹', '比', '较', '固', '定', '，', '主', '要', '在', '家', '从', '事', '网', '络', '销', '售', '手', '机',
               '业', '务', '，', '偶', '尔', '外', '出', '超', '市', '购', '买', '日', '用', '品', '，', '自', '述', '外', '出', '史', '均',
               '有', '佩', '戴', '一', '次', '性', '口', '罩', '。', '2', '0', '2', '0', '年', '1', '0', '月', '2', '4', '日', '两',
               '病', '例', '在', '土', '耳', '其', '伊', '斯', '坦', '布', '尔', '航', '空', '公', '司', '指', '定', '的', '医', '疗', '机',
               '构', '（', 'r', 'e', 'p', 'u', 'b', 'l', 'i', 'c', 'o', 'f', 't', 'u', 'r', 'k', 'e', 'y', 'i', 's', 't',
               'a', 'n', 'b', 'u', 'l', 'k', 'a', 'n', 'u', 'n', 'i', 't', 'i', 'a', 'i', 'n', 'i', 'n', 'g', 'a', 'n',
               'd', 'r', 'e', 's', 'e', 'a', 'r', 'c', 'h', 'h', 'o', 's', 'p', 'i', 't', 'a', 'l', '）', '采', '咽', '拭',
               '子', '新', '冠', '核', '酸', '检', '测', '阴', '性', '。', '1', '0', '月', '2', '7', '日', '下', '午', '1', '5', '时',
               '由', '家', '人', '（', '名', '字', '拒', '绝', '提', '供', '）', '开', '车', '送', '两', '人', '前', '往', '土', '耳', '其',
               '伊', '斯', '坦', '布', '尔', '机', '场', '，', '当', '地', '时', '间', '1', '0', '月', '2', '7', '日', '1', '8', '时',
               '3', '5', '分', '乘', '坐', 't', 'k', '7', '2', '航', '班', '（', '座', '位', '号', '1', '4', 'c', '、', '1', '4',
               'd', '）', '，', '北', '京', '时', '间', '1', '0', '月', '2', '8', '日', '上', '午', '9', '：', '4', '0', '到', '达',
               '广', '州', '白', '云', '国', '际', '机', '场', '。', '配', '合', '广', '州', '海', '关', '闭', '环', '管', '理', '检', '疫',
               '采', '样', '后', '于', '下', '午', '1', '5', '时', '左', '右', '到', '达', '白', '云', '区', '京', '溪', '维', '也', '纳',
               '酒', '店', '（', 'm', 'o', 'h', 'd', 'y', 'a', 's', 's', 'i', 'n', 'h', 'a', 'j', 'i', 'l', 'a', 'c', 'h',
               'w', 'a', 'r', '房', '间', '号', '：', '8', '3', '1', '3', '房', '，', 'f', 'u', 's', 'h', 'a', 'n', 'j', 'a',
               'b', 'd', 'u', 'l', 'h', 'a', 'd', 'i', '房', '间', '号', '：', '8', '3', '1', '2', '房', '）', '隔', '离', '观',
               '察', '，', '无', '再', '外', '出', '，', '期', '间', '无', '互', '相', '串', '门', '情', '况', '。', '（', '二', '）', '暴',
               '露', '史', '两', '名', '病', '例', '自', '述', '常', '住', '土', '耳', '其', '伊', '斯', '坦', '布', '尔', '，', '一', '个',
               '无', '业', '在', '家', '，', '另', '一']
    goldenlabel= ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-port', 'I-port',
            'I-port', 'I-port', 'I-port', 'I-port', 'I-port', 'I-port', 'I-port', 'I-port', 'I-port', 'I-port', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-port', 'I-port', 'I-port', 'I-port', 'I-port',
            'I-port', 'I-port', 'I-port', 'I-port', 'I-port', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    modelpred= ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-abroad', 'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad',
           'I-abroad', 'I-abroad', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-abroad', 'I-abroad',
           'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad',
           'I-abroad', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-abroad', 'I-abroad', 'I-abroad',
           'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad', 'I-abroad', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-insulate',
           'I-insulate', 'I-insulate', 'I-insulate', 'I-insulate', 'I-insulate', 'I-insulate', 'I-insulate',
           'I-insulate', 'I-insulate', 'I-insulate', 'I-insulate', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

    labels = ['name', 'workplace', 'occupation', 'native', 'abroad',
              'domestic', 'port', 'insulate', 'hospital', 'date',
              'person', 'location', 'mask', 'airline', 'license',
              'symptom', 'test', 'IG', 'CT', 'blood']

    map_dict = dict()
    for label in labels:
        map_dict[label] = []


    entity_name = ""
    flag, sentences = [], []
    sentences.append(''.join(sentence))

    for char, tag in zip(sentences[0], modelpred):
        if tag[0] == "B":
            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]

                for label in labels:
                    if y[0] in label:
                        map_dict[label].append(entity_name)
                flag.clear()
                entity_name = ""
            entity_name += char
            flag.append(tag[2:])
        elif tag[0] == "I":
            entity_name += char
            flag.append(tag[2:])
        else:
            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]

                for label in labels:
                    if y[0] in label:
                        map_dict[label].append(entity_name)
                flag.clear()
            flag.clear()
            entity_name = ""

    print("abroad: ", map_dict['abroad'])
    print("insulate: ", map_dict['insulate'])


    # gword = []
    # pword = []
    # for i in range(len(sentence)):
    #     if modelpred[i] != 'O':
    #         pword.append(sentence[i])
        # if goldenlabel[i] != 'O':
        #     gword.append(sentence[i])
    # print("gword: ", gword)
    # print("pword: ", pword)

    # for j in range(len(sents)):
    #     word_s = []
    #     sent = sents[j]
    #     for i in range(len(sent)):
    #         if y_p[j][i] != 'O':
    #             word_s.append(sent[i])
    #     word.append(word_s)
    # print(word)