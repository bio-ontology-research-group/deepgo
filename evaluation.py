#!/usr/bin/env python



import os
import numpy as np
import click as ck
import pandas as pd
from utils import get_gene_ontology, get_go_set, get_anchestors, FUNC_DICT
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

DATA_ROOT = 'data/swiss/'

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
    # FFPred preds
    preds_dict = {}
    # files = os.listdir('data/ffpred/')
    # for fl in files:
    # with open('data/gofdr/predictions.tab') as f:
    #     for line in f:
    #         it = line.strip().split('\t')
    #         target_id = it[0]
    #         if function[1].upper() != it[2]:
    #             continue
    #         if target_id not in preds_dict:
    #             preds_dict[target_id] = list()
    #         preds_dict[target_id].append((it[1], float(it[3])))
    # print(len(preds_dict))
    target_ids = list()
    predictions = list()
    for key, val in preds_dict.items():
        target_ids.append(key)
        predictions.append(val)
    # pred_df = pd.DataFrame({'targets': target_ids, 'predictions': predictions})

    targets = dict()
    with open('data/cafa3/CAFA3_benchmark20170605/groundtruth/leafonly_' + function.upper() +'O_unique.txt') as f:
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
    for target, gos in targets.items():
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
    df.to_pickle(DATA_ROOT + 'model_preds_filtered_' + function + '.pkl')
    
    def reshape(values):
        values = np.hstack(values).reshape(
            len(values), len(values[0]))
        return values

    preds = reshape(df['predictions'].values)
    labels = reshape(df['labels'].values)
    # preds = df['predictions'].values
    gos = df['gos'].values
    f, p, r, t, preds_max = compute_performance(preds, labels, gos)
    print(f, p, r)
    # labels = list()
    # scores = list()
    # for i in range(len(preds)):
    #     all_gos = set()
    #     for go_id in gos[i]:
    #         if go_id in all_functions:
    #             all_gos |= get_anchestors(go, go_id)
    #     all_gos.discard(GO_ID)
    #     scores_dict = {}
    #     for val in preds[i]:
    #         go_id, score = val
    #         if go_id in all_functions:
    #             go_set = get_anchestors(go, go_id)
    #             for g_id in go_set:
    #                 if g_id not in scores_dict or scores_dict[g_id] < score:
    #                     scores_dict[g_id] = score
    #     all_preds = set(scores_dict) # | all_gos
    #     all_preds.discard(GO_ID)
    #     for go_id in all_preds:
    #         if go_id in scores_dict:
    #             scores.append(scores_dict[go_id])
    #         else:
    #             scores.append(0)
    #         if go_id in all_gos:
    #             labels.append(1)
    #         else:
    #             labels.append(0)
        
    # scores = np.array(scores)
    # labels = np.array(labels)
    roc_auc = compute_roc(preds, labels)
    print(roc_auc)
    # preds_max = (scores > t).astype(np.int32)
    mcc = compute_mcc(preds_max, labels)
    print(mcc)


def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_mcc(preds, labels):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

    
def compute_performance(preds, labels, gos):
    # preds = np.round(preds, decimals=2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        # predictions = list()
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(preds.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            all_gos = set()
            all_preds = set()
            for go_id in gos[i]:
                if go_id in all_functions:
                    all_gos |= get_anchestors(go, go_id)
            all_gos.discard(GO_ID)
            # for val in preds[i]:
            #     go_id, score = val
            #     if score > threshold and go_id in all_functions:
            #         all_preds |= get_anchestors(go, go_id)
            # all_preds.discard(GO_ID)
            # predictions.append(all_preds)
            # tp = len(all_gos.intersection(all_preds))
            # fp = len(all_preds) - tp
            # fn = len(all_gos) - tp
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
