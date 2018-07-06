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
from aaindex import is_ok, AAINDEX
import click as ck
from subprocess import Popen, PIPE

DATA_ROOT = 'data/binary/'
MAXLEN = 2000

@ck.command()
def main():
    global go
    go = get_gene_ontology(DATA_ROOT + 'go.obo', with_rels=True)
    func_df = pd.read_pickle(DATA_ROOT + 'functions.pkl')
    global functions
    functions = func_df['functions'].values
    global func_set
    func_set = set(functions)
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
    print('Gram length:', gram_len)
    print('Vocabulary size:', len(vocab))
    ngrams = list()
    df = pd.read_pickle(DATA_ROOT + 'swissprot_exp.pkl')
    # Filtering data by sequences
    index = list()
    for i, row in df.iterrows():
        seq = row['sequences']
        if is_ok(seq) and len(seq) <= MAXLEN:
            index.append(i)
    df = df.loc[index]

    for i, row in df.iterrows():
        seq = row['sequences']
        grams = np.zeros((len(seq) - gram_len + 1, ), dtype='int32')
        for i in range(len(seq) - gram_len + 1):
            grams[i] = vocab[seq[i: (i + gram_len)]]
        ngrams.append(grams)
    res_df = pd.DataFrame({
        'accessions': df['accessions'],
        'proteins': df['proteins'],
        'ngrams': ngrams,
        'functions': df['functions'],
        'sequences': df['sequences']})
    print(len(res_df))
    return res_df


def load_rep_df():
    df = pd.read_pickle('data/graph_new_embeddings_proteins.pkl')
    return df


def load_embeds():
    df = pd.read_pickle('data/graph_new_embeddings.pkl')
    embeds = {}
    for row in df.itertuples():
        embeds[row.accessions] = row.embeddings
    return embeds


def load_org_df():
    df = pd.read_pickle('data/protein_orgs.pkl')
    return df

def load_interpros():
    proteins = list()
    interpros = list()
    with open('data/latest/interpros.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            if len(it) > 1:
                proteins.append(it[0])
                interpros.append(it[1:])
    df = pd.DataFrame({'proteins': proteins, 'interpros': interpros})
    return df

def run(*args, **kwargs):
    # df = load_data()
    # org_df = load_org_df()
    # rep_df = load_rep_df()
    # ipro_df = load_interpros()
    # df = pd.merge(df, org_df, on='proteins', how='left')
    # df = pd.merge(df, rep_df, on='proteins', how='left')
    # df = pd.merge(df, ipro_df, on='proteins', how='left')
    # embeds = load_embeds()
    # mapping = {}
    # with open(DATA_ROOT + 'noembed.map') as f:
    #     for line in f:
    #         it = line.strip().split('\t')
    #         mapping[it[0]] = embeds[it[1]]
    # f = open(DATA_ROOT + 'noembed.fasta', 'w')
    # for i, row in df.iterrows():
    #     if not isinstance(row['embeddings'], np.ndarray):
    #         prot_id = row['proteins']
    #         if prot_id in mapping:
    #             df.at[i, 'embeddings'] = mapping[prot_id]
    #         else:
    #             f.write(('>' + prot_id + '\n' + row['sequences'] + '\n'))

    # #df = df[df['orgs'] == '10090']
    # print(len(df))
    df = pd.read_pickle(DATA_ROOT + 'data.pkl')
    index = np.arange(len(df))
    train_n = int(len(df) * 0.8)
    train_df = df.iloc[index[:train_n]]
    train_df.to_pickle(DATA_ROOT + 'train.pkl')
    test_df = df.iloc[index[train_n:]]
    test_df.to_pickle(DATA_ROOT + 'test.pkl')
    func_df = pd.read_pickle(DATA_ROOT + 'functions.pkl')
    functions = func_df['functions']
    for go_id in functions:
        positives = list()
        negatives = list()
        for i, row in train_df.iterrows():
            if go_id in row['functions']:
                positives.append(i)
            else:
                negatives.append(i)
        np.random.shuffle(negatives)
        negatives = negatives[:len(positives)]
        index = positives + negatives
        np.random.shuffle(index)
        dt = train_df.loc[index]
        dt.to_pickle(DATA_ROOT + go_id.replace(':', '_') + '_train.pkl')
        print('train', go_id, len(dt))

        positives = list()
        negatives = list()
        for i, row in test_df.iterrows():
            if go_id in row['functions']:
                positives.append(i)
            else:
                negatives.append(i)
        np.random.shuffle(negatives)
        negatives = negatives[:len(positives)]
        index = positives + negatives
        np.random.shuffle(index)
        dt = test_df.loc[index]
        dt.to_pickle(DATA_ROOT + go_id.replace(':', '_') + '_test.pkl')
        print('test', go_id, len(dt))


if __name__ == '__main__':
    main()
