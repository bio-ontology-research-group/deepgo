#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_go_set,
    MOLECULAR_FUNCTION,
    BIOLOGICAL_PROCESS,
    CELLULAR_COMPONENT)

DATA_ROOT = 'data/swiss/'

GO_ID = MOLECULAR_FUNCTION

go = get_gene_ontology('go.obo')

func_df = pd.read_pickle(DATA_ROOT + 'mf.pkl')
functions = func_df['functions'].values
func_set = set(functions)


def compute_total_performance():
    fs = 0.0
    n = 0
    t = 0.5
    test_df = pd.read_pickle(DATA_ROOT + 'test-mf.pkl')
    all_functions = get_go_set(go, GO_ID)
    with open(DATA_ROOT + 'predictions-mf.txt', 'r') as f:
        l = 0
        for line in f:
            gos = test_df['gos'][l]
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
    with open(DATA_ROOT + 'predictions-mf.txt', 'r') as f:
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
    compute_performance()

if __name__ == '__main__':
    main(*sys.argv)
