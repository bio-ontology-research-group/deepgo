#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    FUNC_DICT,
    EXP_CODES)
from aaindex import is_ok
import click as ck

DATA_ROOT = 'data/swiss/'


@ck.command()
@ck.option(
    '--function',
    default='mf',
    help='Function (mf, bp, cc)')
@ck.option(
    '--split',
    default=0.8,
    help='Train test split')
def main(function, split):
    global SPLIT
    SPLIT = split
    global GO_ID
    GO_ID = FUNC_DICT[function]
    global go
    go = get_gene_ontology('go.obo')
    global FUNCTION
    FUNCTION = function
    func_df = pd.read_pickle(DATA_ROOT + FUNCTION + '.pkl')
    global functions
    functions = func_df['functions'].values
    global func_set
    func_set = get_go_set(go, GO_ID)
    print(len(functions))
    global go_indexes
    go_indexes = dict()
    for ind, go_id in enumerate(functions):
        go_indexes[go_id] = ind
    run()


def load_data():
    ngram_df = pd.read_pickle(DATA_ROOT + 'ngrams.pkl')
    vocab = {}
    for key, gram in enumerate(ngram_df['ngrams']):
        vocab[gram] = key + 1
    gram_len = len(ngram_df['ngrams'][0])
    print(('Gram length:', gram_len))
    print(('Vocabulary size:', len(vocab)))
    proteins = list()
    gos = list()
    labels = list()
    ngrams = list()
    sequences = list()
    accessions = list()
    df = pd.read_pickle(DATA_ROOT + 'swissprot_exp.pkl')
    # Filtering data by sequences
    index = list()
    for i, row in df.iterrows():
        if is_ok(row['sequences']):
            index.append(i)
    df = df.loc[index]

    for i, row in df.iterrows():
        go_list = []
        for item in row['annots']:
            items = item.split('|')
            if items[1] in EXP_CODES:
                go_list.append(items[0])
            # go_list.append(items[0])
        go_set = set()
        for go_id in go_list:
            if go_id in func_set:
                go_set |= get_anchestors(go, go_id)
        if not go_set or GO_ID not in go_set:
            continue
        go_set.remove(GO_ID)
        gos.append(go_list)
        proteins.append(row['proteins'])
        accessions.append(row['accessions'])
        seq = row['sequences']
        sequences.append(seq)
        grams = np.zeros((len(seq) - gram_len + 1, ), dtype='int32')
        for i in range(len(seq) - gram_len + 1):
            grams[i] = vocab[seq[i: (i + gram_len)]]
        ngrams.append(grams)
        label = np.zeros((len(functions),), dtype='int32')
        for go_id in go_set:
            if go_id in go_indexes:
                label[go_indexes[go_id]] = 1
        labels.append(label)
    res_df = pd.DataFrame({
        'accessions': accessions,
        'proteins': proteins,
        'ngrams': ngrams,
        'labels': labels,
        'gos': gos,
        'sequences': sequences})
    print((len(res_df)))
    return res_df


def load_rep_df():
    df = pd.read_pickle('data/graph_new_embeddings.pkl')
    return df


def load_org_df():
    df = pd.read_pickle('data/protein_orgs.pkl')
    return df


def run(*args, **kwargs):
    df = load_data()
    org_df = load_org_df()
    rep_df = load_rep_df()
    df = pd.merge(df, org_df, on='proteins', how='left')
    df = pd.merge(df, rep_df, on='accessions', how='left')
    missing_rep = 0
    for i, row in df.iterrows():
        if not isinstance(row['embeddings'], np.ndarray):
            row['embeddings'] = np.zeros((256,), dtype='float32')
            missing_rep += 1
    print(('Missing network reps:', missing_rep))
    df = df[df['orgs'] == '9606']
    # index = df.index.values
    # np.random.seed(seed=0)
    # np.random.shuffle(index)
    # train_n = int(len(df) * SPLIT)
    # train_df = df.loc[index[:train_n]]
    # test_df = df.loc[index[train_n:]]
    # prots_df = pd.read_pickle('data/swiss/clusters.pkl')
    # train_df = df[df['proteins'].isin(prots_df['proteins'])]
    # test_df = df[~df['proteins'].isin(prots_df['proteins'])]
    # print(len(train_df), len(test_df))
    # train_df.to_pickle(DATA_ROOT + 'train-' + FUNCTION + '.pkl')
    # test_df.to_pickle(DATA_ROOT + 'test-' + FUNCTION + '.pkl')


if __name__ == '__main__':
    main()
