#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    MOLECULAR_FUNCTION,
    BIOLOGICAL_PROCESS,
    CELLULAR_COMPONENT)

DATA_ROOT = 'data/'

GO_ID = BIOLOGICAL_PROCESS

# go = get_gene_ontology('go.obo')

# func_df = pd.read_pickle(DATA_ROOT + 'bp.pkl')
# functions = func_df['functions'].values
# func_set = set(functions)


def filter_interactions():
    with open(DATA_ROOT + 'protein.links.v10.txt', 'r') as f:
        with open(DATA_ROOT + 'interactions.txt', 'w') as w:
            next(f)
            for line in f:
                items = line.strip().split(' ')
                score = float(items[2])
                if score >= 0.7:
                    w.write(line)


def main(*args, **kwargs):
    filter_interactions()

if __name__ == '__main__':
    main(*sys.argv)
