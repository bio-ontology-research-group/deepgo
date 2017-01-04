#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_go_set,
    get_go_sets,
    get_anchestors,
    BIOLOGICAL_PROCESS,
    MOLECULAR_FUNCTION,
    CELLULAR_COMPONENT,
    FUNC_DICT)
from multiprocessing import Pool
from collections import deque


DATA_ROOT = 'data/cafa3/'
ORG = ''
FILENAME = 'data' + ORG + '.txt'
ANNOT_NUM = 50
FUNCTION = 'mf'

GO_ID = FUNC_DICT[FUNCTION]
FUNCTION += ORG

go = get_gene_ontology('go.obo')

functions = deque()


# Add functions to deque in topological order
def dfs(go_id):
    if go_id not in functions:
        for ch_id in go[go_id]['children']:
            dfs(ch_id)
        functions.append(go_id)


dfs(GO_ID)
functions.remove(GO_ID)
functions.reverse()
functions = list(functions)
print(len(functions))
func_set = set(functions)
go_indexes = dict()
for ind, go_id in enumerate(functions):
    go_indexes[go_id] = ind


def get_functions():
    annots = dict()
    with open(DATA_ROOT + FILENAME, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            go_set = set()
            for go_id in items[2].split('; '):
                if go_id in func_set:
                    go_set |= get_anchestors(go, go_id)
            for go_id in go_set:
                if go_id not in annots:
                    annots[go_id] = 0
                annots[go_id] += 1
    filtered = list()
    for go_id in functions:
        if go_id in annots and annots[go_id] >= ANNOT_NUM:
            filtered.append(go_id)
    print len(filtered)
    df = pd.DataFrame({'functions': filtered})
    df.to_pickle(DATA_ROOT + FUNCTION + '.pkl')
    print 'Saved ' + DATA_ROOT + FUNCTION + '.pkl'


def main(*args, **kwargs):
    get_functions()


if __name__ == '__main__':
    main(*sys.argv)
