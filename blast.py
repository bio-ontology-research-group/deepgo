#!/usr/bin/env python
from __future__ import print_function
import click as ck
import pandas as pd
import numpy as np
from utils import (
    BIOLOGICAL_PROCESS, MOLECULAR_FUNCTION, CELLULAR_COMPONENT,
    EXP_CODES, get_anchestors, get_gene_ontology, get_go_set)

DATA_ROOT = 'data/swissexp/'

@ck.command()
def main():

    # convert()
    f, p, r = compute_performance('bp')
    print(f, p, r)
    f, p, r = compute_performance('mf')
    print(f, p, r)
    f, p, r = compute_performance('cc')
    print(f, p, r)


def compute_performance(func):
    go = get_gene_ontology()
    train_df = pd.read_pickle('data/swissexp/train-' + func + '.pkl')
    test_df = pd.read_pickle('data/swissexp/test-' + func + '.pkl')

    train_labels = {}
    test_labels = {}
    for i, row in train_df.iterrows():
        go_set = set()
        for go_id in row['gos']:
            if go_id in go:
                go_set |= get_anchestors(go, go_id)
        train_labels[row['proteins']] = go_set

    for i, row in test_df.iterrows():
        go_set = set()
        for go_id in row['gos']:
            if go_id in go:
                go_set |= get_anchestors(go, go_id)
        test_labels[row['proteins']] = go_set

    preds = list()
    test = list()
    with open('data/swissexp/blast-' + func + '.res') as f:
        for line in f:
            it = line.strip().split('\t')
            preds.append(train_labels[it[1]])
            test.append(test_labels[it[0]])

    total = 0
    p = 0.0
    r = 0.0
    f = 0.0
    p_total = 0
    for label, pred in zip(test, preds):
        # tp = np.sum(label * pred)
        # fp = np.sum(pred) - tp
        # fn = np.sum(label) - tp
        tp = len(label.intersection(pred))
        fp = len(pred) - tp
        fn = len(label) - tp

        if tp == 0 and fp == 0 and fn == 0:
            continue
        total += 1
        if tp != 0:
            p_total += 1
            precision = tp / (1.0 * (tp + fp))
            recall = tp / (1.0 * (tp + fn))
            p += precision
            r += recall
    p /= p_total
    r /= total
    f = 2 * p * r / (p + r)
    return f, p, r


def convert():
    df = pd.read_pickle(DATA_ROOT + 'train-bp.pkl')
    with open(DATA_ROOT + 'train-bp.fa', 'w') as f:
        for i, row in df.iterrows():
            f.write('>' + row['proteins'] + '\n')
            f.write(to_fasta(str(row['sequences'])))


def to_fasta(sequence):
    length = 60
    n = len(sequence)
    res = ''
    for i in xrange(0, n, length):
        res += sequence[i: i + length] + '\n'
    return res


if __name__ == '__main__':
    main()
