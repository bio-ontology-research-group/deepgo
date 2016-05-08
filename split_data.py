#!/usr/bin/env python
import sys
import os
from collections import deque
import time
from utils import shuffle, get_gene_ontology
from aaindex import INVALID_ACIDS


DATA_ROOT = 'data/'
RESULT_ROOT = 'data/swiss/'
FILES = (
    'uniprot-swiss-bp.txt',)


MIN_LEN = 24


def is_ok(seq):
    if len(seq) < MIN_LEN:
        return False
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def load_all_proteins():
    prots = list()
    for i in range(len(FILES)):
        file_name = FILES[i]
        with open(DATA_ROOT + file_name, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                prot_id = line[0]
                seq = line[1]
                gos = line[2]
                if is_ok(seq):
                    prots.append((prot_id, seq, gos))
    return prots


def main():
    start_time = time.time()
    print 'Loading all proteins'
    all_prots = load_all_proteins()
    shuffle(all_prots)
    split = 0.8
    train_len = int(len(all_prots) * split)

    with open(RESULT_ROOT + 'train-bp.txt', 'w') as f:
        for prot_id, seq, gos in all_prots[:train_len]:
            f.write(prot_id + '\t' + seq + '\t' + gos + '\n')
    with open(RESULT_ROOT + 'test-bp.txt', 'w') as f:
        for prot_id, seq, gos in all_prots[train_len:]:
            f.write(prot_id + '\t' + seq + '\t' + gos + '\n')

    end_time = time.time() - start_time
    print 'Done in %d seconds' % (end_time, )

if __name__ == '__main__':
    main()
