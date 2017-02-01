#!/usr/bin/env python
from __future__ import print_function
import click as ck
import pandas as pd
from utils import (
    BIOLOGICAL_PROCESS, MOLECULAR_FUNCTION, CELLULAR_COMPONENT,
    EXP_CODES, get_anchestors, get_gene_ontology, get_go_set)


@ck.command()
def main():
    f, p, r = compute_performance()
    print(f, p, r)


def compute_performance():
    go = get_gene_ontology()
    go_set = get_go_set(go, CELLULAR_COMPONENT)
    df = pd.read_pickle('data/cafa3/swissprot_exp.pkl')
    annots = {}
    for i, row in df.iterrows():
        annots[row['proteins']] = set()
        for go_id in row['annots']:
            go_id = go_id.split('|')
            if go_id[1] in EXP_CODES and go_id[0] in go_set:
                annots[row['proteins']] |= get_anchestors(go, go_id[0])

        annots[row['proteins']].discard(BIOLOGICAL_PROCESS)
        annots[row['proteins']].discard(MOLECULAR_FUNCTION)
        annots[row['proteins']].discard(CELLULAR_COMPONENT)

    pred_mapping = dict()
    with open('data/blast_uniq.tab') as f:
        for line in f:
            items = line.strip().split('\t')
            pred_mapping[items[0]] = items[1]
    total = 0
    p = 0.0
    r = 0.0
    f = 0.0
    for prot, pred_prot in pred_mapping.iteritems():
        real_annots = annots[prot]
        pred_annots = annots[pred_prot]
        tp = len(real_annots.intersection(pred_annots))
        fp = len(pred_annots - real_annots)
        fn = len(real_annots - pred_annots)
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
    df = pd.read_pickle('data/cafa3/train.pkl')
    with open('data/cafa3/train.fa', 'w') as f:
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
