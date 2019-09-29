#!/usr/bin/env python

import click as ck
import pandas as pd
import numpy as np
from utils import (
    BIOLOGICAL_PROCESS, MOLECULAR_FUNCTION, CELLULAR_COMPONENT,
    EXP_CODES, get_anchestors, get_gene_ontology, get_go_set)

DATA_ROOT = 'data/swiss/'


@ck.command()
@ck.option('--function', default='mf', help='Function')
def main(function):
    # fill_missing(function)
    # f, p, r = compute_performance('bp')
    # print('%.3f & %.3f & %.3f' % (f, p, r))
    # f, p, r = compute_performance('mf')
    # print('%.3f & %.3f & %.3f' % (f, p, r))
    # f, p, r = compute_performance('cc')
    # print('%.3f & %.3f & %.3f' % (f, p, r))
    convert('')


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
        train_labels[row['proteins']] = row['labels']

    for i, row in test_df.iterrows():
        go_set = set()
        for go_id in row['gos']:
            if go_id in go:
                go_set |= get_anchestors(go, go_id)
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
    p_total = 0
    for label, pred in zip(test, preds):
        tp = np.sum(label * pred)
        fp = np.sum(pred) - tp
        fn = np.sum(label) - tp
        # tp = len(label.intersection(pred))
        # fp = len(pred) - tp
        # fn = len(label) - tp

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


def convert(function):
    df = pd.read_pickle('data/' + 'sequence_embeddings.pkl')
    f1 = open(DATA_ROOT + 'embeddings.fa', 'w')
    # f2 = open(DATA_ROOT + 'test-missing.fa', 'w')
    seqs = set()
    for i, row in df.iterrows():
        # missing = np.sum(row['embeddings']) == 0
        # if not missing:
        seq = row['sequences']
        if seq not in seqs:
            seqs.add(seq)
            f1.write('>' + row['accessions'] + '\n')
            f1.write(to_fasta(str(seq)))
        #else:
        #    f2.write('>' + row['proteins'] + '\n')
        #    f2.write(to_fasta(str(row['sequences'])))
    f1.close()
    #f2.close()


def to_fasta(sequence):
    length = 60
    n = len(sequence)
    res = ''
    for i in range(0, n, length):
        res += sequence[i: i + length] + '\n'
    return res


def fill_missing(function):
    tt = 'train'
    df = pd.read_pickle(DATA_ROOT + tt + '-' + function + '.pkl')
    mapping = dict()
    with open(DATA_ROOT + 'blast-' + tt + '-cc.res') as f:
        for line in f:
            it = line.strip().split('\t')
            mapping[it[0]] = it[1]
    embeddings = dict()
    for i, row in df.iterrows():
        missing = np.sum(row['embeddings']) == 0
        if not missing:
            embeddings[row['proteins']] = row['embeddings']

    m = 0
    for i, row in df.iterrows():
        missing = np.sum(row['embeddings']) == 0
        if missing and row['proteins'] in mapping:
            row['embeddings'] = embeddings[mapping[row['proteins']]]
            m += 1
    print(m)
    n = 0
    for i, row in df.iterrows():
        missing = np.sum(row['embeddings']) == 0
        if missing:
            n += 1
    print(n)
    df.to_pickle(DATA_ROOT + tt + '-' + function + '-nomissing.pkl')


if __name__ == '__main__':
    main()
