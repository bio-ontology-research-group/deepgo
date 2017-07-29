#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import click as ck
import pandas as pd
from utils import get_gene_ontology, get_go_set, get_anchestors, FUNC_DICT
from sklearn.metrics import roc_curve, auc

DATA_ROOT = 'data/cafa3/done3/'

@ck.command()
@ck.option('--function', default='mf', help='Function')
def main(function):
    global go
    go = get_gene_ontology()
    func_df = pd.read_pickle(DATA_ROOT + function + '.pkl')
    global functions
    functions = func_df['functions'].values
    func_index = dict()
    for i, go_id in enumerate(functions):
        func_index[go_id] = i
    global func_set
    func_set = set(func_index)
    global GO_ID
    GO_ID = FUNC_DICT[function]
    global all_functions
    all_functions = get_go_set(go, GO_ID)
    pred_df = pd.read_pickle(DATA_ROOT + 'model_preds_' + function + '.pkl')
    targets = dict()
    with open('data/cafa3/CAFA3_benchmark20170605/groundtruth/leafonly_MFO_unique.txt') as f:
        for line in f:
            it = line.strip().split('\t')
            target = it[0]
            go_id = it[1]
            if target not in targets:
                targets[target] = list()
            targets[target].append(go_id)
    target_ids = list()
    labels = list()
    go_ids = list()
    for target, gos in targets.iteritems():
        go_set = set()
        for go_id in gos:
            if go_id in all_functions:
                go_set |= get_anchestors(go, go_id)
        label = np.zeros((len(functions),), dtype=np.int32)
        for go_id in go_set:
            if go_id in func_index:
                label[func_index[go_id]] = 1
        target_ids.append(target)
        go_ids.append(go_set)
        labels.append(label)
    df = pd.DataFrame({'targets': target_ids, 'gos': go_ids, 'labels': labels})
    df = pd.merge(df, pred_df, on='targets', how='inner')

    def reshape(values):
        values = np.hstack(values).reshape(
            len(values), len(values[0]))
        return values

    preds = reshape(df['predictions'].values)
    labels = reshape(df['labels'].values)
    gos = df['gos'].values
    f, p, r, t, preds_max = compute_performance(preds, labels, gos)
    roc_auc = compute_roc(preds, labels)
    print(roc_auc)
    print(f, p, r)


def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

    
def compute_performance(preds, labels, gos):
    preds = np.round(preds, decimals=2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in xrange(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            all_gos = set()
            for go_id in gos[i]:
                if go_id in all_functions:
                    all_gos |= get_anchestors(go, go_id)
            all_gos.discard(GO_ID)
            all_gos -= func_set
            fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if total > 0 and p_total > 0:
            r /= total
            p /= p_total
            if p + r > 0:
                f = 2 * p * r / (p + r)
                if f_max < f:
                    f_max = f
                    p_max = p
                    r_max = r
                    t_max = threshold
                    predictions_max = predictions

    return f_max, p_max, r_max, t_max, predictions_max



if __name__ == '__main__':
    main()
