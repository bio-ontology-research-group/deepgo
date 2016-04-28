#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors)


DATA_ROOT = 'data/swiss/'
FILENAME = 'uniprot-swiss-mol-func.txt'


go = get_gene_ontology('go.obo')
functions = get_go_set(go, 'GO:0003674')


def load_data():
    proteins = list()
    sequences = list()
    gos = list()
    with open(DATA_ROOT + FILENAME, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            proteins.append(items[0])
            sequences.append(items[1])
            go_set = set()
            for go_id in items[2].split('; '):
                if go_id in functions:
                    go_set |= get_anchestors(go, go_id)
            go_set.remove('GO:0003674')
            gos.append(list(go_set))
    return proteins, sequences, gos


def main(*args, **kwargs):
    proteins, sequences, gos = load_data()
    data = {
        'proteins': np.array(proteins),
        'sequences': np.array(sequences),
        'gos': np.array(gos)}
    df = pd.DataFrame(data)
    df.to_pickle(DATA_ROOT + 'test.pkl')

if __name__ == '__main__':
    main(*sys.argv)
