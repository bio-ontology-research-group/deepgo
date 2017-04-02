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
    f, p, r = compute_performance('mf')
    print(f, p, r)
    f, p, r = compute_performance('cc')
    print(f, p, r)
    f, p, r = compute_performance('bp')
    print(f, p, r)


def compute_performance(func):
    go = get_gene_ontology()
    go_set = get_go_set(go, BIOLOGICAL_PROCESS)
    train_df = pd.read_pickle('data/swissexp/train-' + func + '.pkl')
    test_df = pd.read_pickle('data/swissexp/test-' + func + '.pkl')

    train_labels = {}
    test_labels = {}
    for i, row in train_df.iterrows():
        train_labels[row['proteins']] = row['labels']

    for i, row in test_df.iterrows():
        test_labels[row['proteins']] = row['labels']

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
    for label, pred in zip(test, preds):
        tp = np.sum(label * pred)
        fp = np.sum(pred) - tp
        fn = np.sum(label) - tp
        if tp == 0 and fp == 0 and fn == 0:
            continue
        total += 1
        if tp != 0:
            precision = tp / (1.0 * (tp + fp))
            recall = tp / (1.0 * (tp + fn))
            p += precision
            r += recall
            f += 2 * precision * recall / (precision + recall)
    return f / total, p / total, r / total


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
