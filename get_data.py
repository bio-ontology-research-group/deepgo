#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    BIOLOGICAL_PROCESS,
    MOLECULAR_FUNCTION,
    CELLULAR_COMPONENT)
from aaindex import AAINDEX


DATA_ROOT = 'data/swiss/'
FILENAME = 'test-bp.txt'
GO_ID = BIOLOGICAL_PROCESS

go = get_gene_ontology('go.obo')

func_df = pd.read_pickle(DATA_ROOT + 'bp.pkl')
functions = func_df['functions'].values
func_set = set(functions)
print len(functions)
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
            go_set = set()
            for go_id in items[2].split('; '):
                if go_id in func_set:
                    go_set |= get_anchestors(go, go_id)
            if not go_set:
                continue
            go_set.remove(GO_ID)
            gos.append(list(go_set))
            proteins.append(items[0])
            sequences.append(items[1])
            idx = [0] * len(items[1])
            for i in range(len(idx)):
                idx[i] = AAINDEX[items[1][i]]
            indexes.append(idx)
            label = [0] * len(functions)
            for go_id in go_set:
                if go_id in go_indexes:
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
    df.to_pickle(DATA_ROOT + 'test-bp.pkl')

if __name__ == '__main__':
    main(*sys.argv)
