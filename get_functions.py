#!/usr/bin/env python
import pandas as pd
import click as ck
from utils import (
    get_gene_ontology,
    get_anchestors,
    FUNC_DICT,
    EXP_CODES)
from collections import deque
from aaindex import is_ok


DATA_ROOT = 'data/swiss/'


@ck.command()
@ck.option(
    '--function',
    default='mf',
    help='Function (mf, bp, cc)')
@ck.option(
    '--annot-num',
    default=50,
    help='Limit of annotations number for selecting function')
def main(function, annot_num):
    global FUNCTION
    FUNCTION = function
    global GO_ID
    GO_ID = FUNC_DICT[FUNCTION]
    global go
    go = get_gene_ontology('go.obo')
    global functions
    functions = deque()
    dfs(GO_ID)
    functions.remove(GO_ID)
    functions = list(functions)
    print((len(functions)))
    global func_set
    func_set = set(functions)
    global go_indexes
    go_indexes = dict()
    for ind, go_id in enumerate(functions):
        go_indexes[go_id] = ind

    get_functions(annot_num)


# Add functions to deque in topological order
def dfs(go_id):
    if go_id not in functions:
        for ch_id in go[go_id]['children']:
            dfs(ch_id)
        functions.append(go_id)


def get_functions(annot_num):
    df = pd.read_pickle(DATA_ROOT + 'swissprot_exp.pkl')
    annots = dict()
    for i, row in df.iterrows():
        go_set = set()
        if not is_ok(row['sequences']):
            continue
        for go_id in row['annots']:
            go_id = go_id.split('|')
            if go_id[1] not in EXP_CODES:
                continue
            go_id = go_id[0]
            if go_id in func_set:
                go_set |= get_anchestors(go, go_id)
        for go_id in go_set:
            if go_id not in annots:
                annots[go_id] = 0
            annots[go_id] += 1
    filtered = list()
    for go_id in functions:
        if go_id in annots and annots[go_id] >= annot_num:
            filtered.append(go_id)
    print(len(filtered))
    df = pd.DataFrame({'functions': filtered})
    df.to_pickle(DATA_ROOT + FUNCTION + '.pkl')
    print('Saved ' + DATA_ROOT + FUNCTION + '.pkl')


if __name__ == '__main__':
    main()
