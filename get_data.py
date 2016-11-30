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
from text import get_text_reps


FUNCTION = 'bp'
ORG = ''
TT = 'data'

args = sys.argv
if len(args) == 4:
    print args
    TT = args[1]
    if args[2]:
        ORG = '-' + args[2]
    else:
        ORG = ''
    FUNCTION = args[3]

FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

GO_ID = FUNC_DICT[FUNCTION]

DATA_ROOT = 'data/cafa3/'
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
    with open('data/uni_reps.tab', 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            prot_id = it[0]
            rep = np.array(map(float, it[1:]))
            data[prot_id] = rep
    return data


def filter_data():
    prots = set()
    with open('data/uni_reps.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            prots.add(items[0])
    train = list()
    text_reps = get_text_reps()
    with open('data/cafa3/uniprot_sprot.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[0] not in prots:
                train.append(line)
    print(len(train))
    with open('data/cafa3/data.txt', 'w') as f:
        for line in train:
            f.write(line)


def run(*args, **kwargs):
    proteins, sequences, indexes, gos, labels = load_data()
    data = {
        'proteins': proteins,
        'sequences': sequences,
        'indexes': indexes,
        'gos': gos,
        'labels': labels}
    rep = load_rep()
    # text_reps = get_text_reps()
    rep_list = list()
    for prot_id in proteins:
        # text_rep = np.zeros((128,), dtype='float32')
        net_rep = np.zeros((256,), dtype='float32')
        # if prot_id in text_reps:
        #     text_rep = text_reps[prot_id]
        if prot_id in rep:
            net_rep = rep[prot_id]
        rep_list.append(net_rep)
    data['rep'] = rep_list
    df = pd.DataFrame(data)
    df.to_pickle(DATA_ROOT + TT + ORG + '-' + FUNCTION + '.pkl')


def main(*args):
    run()
    # filter_data()


if __name__ == '__main__':
    main(*sys.argv)

