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

FUNCTION = 'bp'
ORG = ''
TT = 'test'

args = sys.argv
if len(args) == 4:
    print args
    TT = args[1]
    ORG = '-' + args[2]
    FUNCTION = args[3]

FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

GO_ID = FUNC_DICT[FUNCTION]

DATA_ROOT = 'data/network/'
FILENAME = TT + '.txt'

go = get_gene_ontology('go.obo')

func_df = pd.read_pickle(DATA_ROOT + FUNCTION + ORG + '.pkl')
functions = func_df['functions'].values
func_set = get_go_set(go, GO_ID)
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
            go_list = items[2].split('; ')
            go_set = set()
            for go_id in go_list:
                if go_id in func_set:
                    go_set |= get_anchestors(go, go_id)
            if not go_set or GO_ID not in go_set:
                continue
            go_set.remove('root')
            go_set.remove(GO_ID)
            gos.append(go_list)
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


def load_rep():
    data = dict()
    with open(DATA_ROOT + 'uni_reps.tab', 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            prot_id = it[0]
            rep = np.array(map(float, it[1:]))
            data[prot_id] = rep
    return data


def filter_data():
    prots = set()
    with open('data/network/uni_reps.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            prots.add(items[0])
    train = list()
    with open('data/network/train.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[0] in prots:
                train.append(line)
    with open('data/network/train.txt', 'w') as f:
        for line in train:
            f.write(line)

    test = list()
    with open('data/network/test.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[0] in prots:
                test.append(line)
    with open('data/network/test.txt', 'w') as f:
        for line in test:
            f.write(line)


def main(*args, **kwargs):
    proteins, sequences, indexes, gos, labels = load_data()
    data = {
        'proteins': proteins,
        'sequences': sequences,
        'indexes': indexes,
        'gos': gos,
        'labels': labels}
    rep = load_rep()
    rep_list = list()
    for prot_id in proteins:
        rep_list.append(rep[prot_id])
    data['rep'] = rep_list
    df = pd.DataFrame(data)
    df.to_pickle(DATA_ROOT + TT + ORG + '-' + FUNCTION + '.pkl')



if __name__ == '__main__':
    main(*sys.argv)

