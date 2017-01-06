#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_anchestors,
    FUNC_DICT,
    EXP_CODES)
from collections import deque


DATA_ROOT = 'data/swissprot/'
ANNOT_NUM = 50
FUNCTION = 'cc'

GO_ID = FUNC_DICT[FUNCTION]

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
    df = pd.read_pickle(DATA_ROOT + 'swissprot_exp.pkl')
    annots = dict()
    for i, row in df.iterrows():
        go_set = set()
        for go_id in row['annots']:
            go_id = go_id.split('|')
            # if go_id[1] not in EXP_CODES:
            #     continue
            go_id = go_id[0]
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
