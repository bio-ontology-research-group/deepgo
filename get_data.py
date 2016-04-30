#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors)
from aaindex import AAINDEX


DATA_ROOT = 'data/yeast/'
FILENAME = 'train.txt'


go = get_gene_ontology('goslim_yeast.obo')
functions = get_go_set(go, 'GO:0003674')
functions.remove('GO:0003674')
functions = list(functions)
go_indexes = dict()
for ind, go_id in enumerate(functions):
    go_indexes[go_id] = ind


def load_data():
    proteins = list()
    sequences = list()
    gos = list()
    labels = list()
    indexes = list()
    with open(DATA_ROOT + FILENAME, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            proteins.append(items[0])
            sequences.append(items[1])
            idx = [0] * len(items[1])
            for i in range(len(idx)):
                idx[i] = AAINDEX[items[1][i]]
            indexes.append(idx)
            go_set = set()
            for go_id in items[2].split('; '):
                if go_id in functions:
                    go_set |= get_anchestors(go, go_id)
            go_set.remove('GO:0003674')
            gos.append(list(go_set))
            label = [0] * len(functions)
            for go_id in go_set:
                label[go_indexes[go_id]] = 1
            labels.append(label)

    return proteins, sequences, indexes, gos, labels


def main(*args, **kwargs):
    proteins, sequences, indexes, gos, labels = load_data()
    data = {
        'proteins': proteins,
        'sequences': sequences,
        'indexes': indexes,
        'gos': gos,
        'labels': labels}
    df = pd.DataFrame(data)
    df.to_pickle(DATA_ROOT + 'train.pkl')

if __name__ == '__main__':
    main(*sys.argv)
