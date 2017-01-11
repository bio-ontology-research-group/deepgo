#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
import pandas as pd
import numpy as np


def to_pickle():
    proteins = list()
    accessions = list()
    sequences = list()
    with open('data/cafa3/uniprot_sprot.tab') as f:
        for line in f:
            items = line.strip().split('\t')
            ids = items[0].split('|')
            proteins.append(ids[2])
            accessions.append(ids[1])
            sequences.append(items[1])
    seq_df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'sequences': sequences
    })
    proteins = list()
    annots = list()
    with open('data/cafa3/swissprot.tab') as f:
        for line in f:
            items = line.strip().split('\t')
            proteins.append(items[0])
            annots.append(items[2:])
    annots_df = pd.DataFrame({
        'proteins': proteins,
        'annots': annots
    })
    df = pd.merge(seq_df, annots_df, on='proteins')
    print(len(df))
    df.to_pickle('data/cafa3/swissprot.pkl')


def filter_exp():
    df = pd.read_pickle('data/cafa3/swissprot.pkl')
    exp_codes = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC'])
    index = list()
    for row in df.iterrows():
        ok = False
        for go_id in row[1]['annots']:
            code = go_id.split('|')[1]
            if code in exp_codes:
                ok = True
                break
        if ok:
            index.append(row[0])
    df = df.loc[index]
    print(len(df))
    df.to_pickle('data/cafa3/swissprot_exp.pkl')


def string_uni():
    mapping = dict()
    with open('data/string2uni.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            mapping[it[0].upper()] = it[1]

    with open('data/string_idmapping.dat') as f:
        for line in f:
            it = line.strip().split('\t')
            mapping[it[2].upper()] = it[0]

    with open('data/uniprot-string.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            mapping[it[1].upper()[:-1]] = it[0]

    embeds = list()
    accessions = list()
    with open('data/graph.mapping.out') as f:
        for line in f:
            it = line.strip().split('\t')
            st_id = it[0].upper()
            if st_id in mapping:
                accessions.append(mapping[st_id])
    # with open('data/graph_reps.tab') as f:
    #     for line in f:
    #         it = line.strip().split('\t')
    #         st_id = it[0].upper()
    #         if st_id in mapping:
    #             accessions.append(mapping[st_id])
    #             rep = np.array(map(float, it[1:]), dtype='float32')
    #             embeds.append(rep)
    df = pd.DataFrame({
        'accessions': accessions})
    print(len(df))
    df.to_pickle('data/graph_accessions.pkl')


def main():
    string_uni()


if __name__ == '__main__':
    main()
