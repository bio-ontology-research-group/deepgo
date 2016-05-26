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
    CELLULAR_COMPONENT)
from aaindex import AAINDEX
from multiprocessing import Pool


DATA_ROOT = 'data/swiss/'
FILENAME = 'train.txt'
ANNOT_NUM = 10
GO_ID = CELLULAR_COMPONENT

go = get_gene_ontology('go.obo')
# functions = get_go_sets(
#     go, [MOLECULAR_FUNCTION, BIOLOGICAL_PROCESS, CELLULAR_COMPONENT])

functions = get_go_set(go, GO_ID)
functions.remove(GO_ID)
functions = list(functions)
func_set = set(functions)
print len(functions)
go_indexes = dict()
for ind, go_id in enumerate(functions):
    go_indexes[go_id] = ind

used = set()
iters = 0
classes = set()


def dfs(go_id):
    used.add(go_id)
    classes.add(go_id)
    global iters
    iters += 1
    for ch_id in go[go_id]['children']:
        if ch_id not in used:
            dfs(ch_id)
        else:
            print 'CYCLE'
    used.remove(go_id)


def filter_functions(go_id):
    if 'annots' in go[go_id] and go[go_id]['annots'] >= ANNOT_NUM:
        return go_id
    return None


def get_functions():
    with open(DATA_ROOT + FILENAME, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            go_set = set()
            for go_id in items[2].split('; '):
                if go_id in func_set:
                    go_set |= get_anchestors(go, go_id)
            for go_id in go_set:
                if 'annots' not in go[go_id]:
                    go[go_id]['annots'] = 0
                go[go_id]['annots'] += 1
    filtered = list()
    pool = Pool(64)
    funcs = pool.map(filter_functions, functions)
    for go_id in funcs:
        if go_id:
            filtered.append(go_id)
    print len(filtered)
    df = pd.DataFrame({'functions': filtered})
    df.to_pickle(DATA_ROOT + 'cc.pkl')


def main(*args, **kwargs):
    get_functions()

if __name__ == '__main__':
    main(*sys.argv)
