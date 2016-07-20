#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    MOLECULAR_FUNCTION,
    BIOLOGICAL_PROCESS,
    CELLULAR_COMPONENT)
from keras.models import model_from_json
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
sys.setrecursionlimit(100000)

DATA_ROOT = 'data/swiss/'
MAXLEN = 1000
GO_ID = BIOLOGICAL_PROCESS

go = get_gene_ontology('go.obo')

func_df = pd.read_pickle(DATA_ROOT + 'cc-human.pkl')
functions = func_df['functions'].values
func_set = set(functions)


def predict():
    test_df = pd.read_pickle(DATA_ROOT + 'train-human-cc.pkl')
    data = test_df['indexes'].values
    data = sequence.pad_sequences(data, maxlen=MAXLEN)
    labels = test_df['labels'].values
    shape = labels.shape
    labels = np.hstack(labels).reshape(shape[0], len(functions))
    labels = labels.transpose()
    batch_size = 128
    logging.info('Loading model')
    with open(DATA_ROOT + 'model_cc_human.json', 'r') as f:
        json_string = next(f)
    model = model_from_json(json_string)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    logging.info('Loading weights')
    model.load_weights(DATA_ROOT + 'hierarchical_cc_human.hdf5')

    predictions = model.predict(
        data, batch_size=batch_size, verbose=1)
    prot_res = list()
    for i in range(len(data)):
        prot_res.append({
            'tp': 0.0, 'fp': 0.0, 'fn': 0.0,
            'pred': list(), 'test': list()})
    for i in range(len(functions)):
        rpred = predictions[i].flatten()
        pred = np.round(rpred)
        test = labels[i]
        for j in range(len(pred)):
            if pred[j] == 1 and test[j] == 1:
                prot_res[j]['tp'] += 1
            elif pred[j] == 1 and test[j] == 0:
                prot_res[j]['fp'] += 1
            elif pred[j] == 0 and test[j] == 1:
                prot_res[j]['fn'] += 1
        logging.info(functions[i])
        logging.info(classification_report(test, pred))
    fs = 0.0
    n = 0
    for prot in prot_res:
        tp = prot['tp']
        fp = prot['fp']
        fn = prot['fn']
        if tp == 0.0 and fp == 0.0 and fn == 0.0:
            continue
        if tp != 0.0:
            recall = tp / (1.0 * (tp + fn))
            precision = tp / (1.0 * (tp + fp))
            fs += 2 * precision * recall / (precision + recall)
        n += 1
    logging.info('Protein centric F measure: \t %f %d' % (fs / n, n))


def compute_total_performance():
    fs = 0.0
    n = 0
    t = 0.5
    test_df = pd.read_pickle(DATA_ROOT + 'test-bp.pkl')
    all_functions = get_go_set(go, GO_ID)
    with open(DATA_ROOT + 'predictions-bp.txt', 'r') as f:
        l = 0
        for line in f:
            gos = test_df['gos'][l]
            go_set = set()
            for go_id in gos:
                if go_id in func_set:
                    go_set |= get_anchestors(go, go_id)
            go_set.remove(GO_ID)
            go_set.remove('root')
            l += 1
            preds = map(float, line.strip().split('\t'))
            tests = map(float, next(f).strip().split('\t'))
            preds = np.array(preds)
            for i in range(len(preds)):
                preds[i] = 0.0 if preds[i] < t else 1.0
            tests = np.array(tests)
            tp = 0.0
            fp = 0.0
            fn = 0.0
            for go_id in gos:
                if go_id not in func_set and go_id in all_functions:
                    fn += 1
            for i in range(len(preds)):
                if tests[i] == 1 and preds[i] == 1:
                    tp += 1
                elif tests[i] == 1 and preds[i] == 0:
                    fn += 1
                elif tests[i] == 0 and preds[i] == 1:
                    fp += 1
            if tp == 0.0 and fp == 0.0 and fn == 0.0:
                continue
            if tp != 0.0:
                recall = tp / (1.0 * (tp + fn))
                precision = tp / (1.0 * (tp + fp))
                fs += 2 * precision * recall / (precision + recall)
            n += 1
    print 'Threshold:', t
    print 'Protein centric F-measure:', fs / n


def compute_performance():
    fs = 0.0
    n = 0
    t = 0.5
    with open(DATA_ROOT + 'predictions-bp.txt', 'r') as f:
        for line in f:
            preds = map(float, line.strip().split('\t'))
            tests = map(float, next(f).strip().split('\t'))
            preds = np.array(preds)
            for i in range(len(preds)):
                preds[i] = 0.0 if preds[i] < t else 1.0
            tests = np.array(tests)
            tp = 0.0
            fp = 0.0
            fn = 0.0
            for i in range(len(preds)):
                if tests[i] == 1 and preds[i] == 1:
                    tp += 1
                elif tests[i] == 1 and preds[i] == 0:
                    fn += 1
                elif tests[i] == 0 and preds[i] == 1:
                    fp += 1
            if tp == 0.0 and fp == 0.0 and fn == 0.0:
                continue
            if tp != 0.0:
                recall = tp / (1.0 * (tp + fn))
                precision = tp / (1.0 * (tp + fp))
                fs += 2 * precision * recall / (precision + recall)
            n += 1
    print 'Threshold:', t
    print 'Protein centric F-measure:', fs / n


def main(*args, **kwargs):
    predict()

if __name__ == '__main__':
    main(*sys.argv)
