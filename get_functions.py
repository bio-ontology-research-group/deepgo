#!/usr/bin/env python
import pandas as pd
import click as ck
from utils import (
    get_gene_ontology,
    get_anchestors,
    get_go_set,
    FUNC_DICT,
    EXP_CODES)
from collections import deque
from aaindex import is_ok


DATA_ROOT = 'data/latest/'


@ck.command()
@ck.option('--annot-num', default=50, help='Annotations')
def main(annot_num):
    global go
    go = get_gene_ontology(DATA_ROOT + 'go.obo', with_rels=True)
    global functions
    functions = deque()
    global func_set
    func_set = set()
    dfs(FUNC_DICT['bp'])
    dfs(FUNC_DICT['mf'])
    dfs(FUNC_DICT['cc'])
    functions.remove(FUNC_DICT['bp'])
    functions.remove(FUNC_DICT['mf'])
    functions.remove(FUNC_DICT['cc'])
    functions = list(functions)
    print(len(functions))

    get_functions(annot_num)


# Add functions to deque in topological order
def dfs(go_id):
    if go_id not in func_set:
        for ch_id in go[go_id]['children']:
            dfs(ch_id)
        functions.append(go_id)
        func_set.add(go_id)

def get_functions(annot_num):
    annots = dict()
    with open(DATA_ROOT + 'annots.tab') as f:
        for line in f:
            go_set = set()
            it = line.strip().split('\t')
            for go_id in it[2:]:
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
    mf = get_go_set(go, FUNC_DICT['mf'])
    bp = get_go_set(go, FUNC_DICT['bp'])
    cc = get_go_set(go, FUNC_DICT['cc'])
    mf_n = 0
    bp_n = 0
    cc_n = 0
    for go_id in filtered:
        if go_id in mf:
            mf_n += 1
        if go_id in bp:
            bp_n += 1
        if go_id in cc:
            cc_n += 1
    print(bp_n, mf_n, cc_n)
    df = pd.DataFrame({'functions': filtered})
    df.to_pickle(DATA_ROOT + 'functions.pkl')
    print('Saved ' + DATA_ROOT + 'functions.pkl')


if __name__ == '__main__':
    main()
