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
    'uniprot-swiss.txt',)


MINLEN = 25
MAXLEN = 1000

ORG_ID = '83333'


def is_ok(seq):
    if len(seq) < MINLEN or len(seq) > MAXLEN:
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


def get_org_prots():
    result = set()
    with open('data/uniprot-all-org.tab', 'r') as f:
        next(f)  # skipping header line
        for line in f:
            items = line.strip().split()
            prot_id = items[0]
            org_id = items[1]
            if org_id == ORG_ID:
                result.add(prot_id)
    return result


def main():
    start_time = time.time()
    print 'Loading all proteins'
    all_prots = load_all_proteins()
    shuffle(all_prots)
    train_prots = list()
    test_prots = list()
    org_prots = get_org_prots()
    for item in all_prots:
        prot_id, seq, gos = item
        if prot_id not in org_prots:
            train_prots.append(item)
        else:
            test_prots.append(item)
    with open(RESULT_ROOT + 'train_ecoli.txt', 'w') as f:
        for prot_id, seq, gos in train_prots:
            f.write(prot_id + '\t' + seq + '\t' + gos + '\n')
    with open(RESULT_ROOT + 'test_ecoli.txt', 'w') as f:
        for prot_id, seq, gos in test_prots:
            f.write(prot_id + '\t' + seq + '\t' + gos + '\n')

    end_time = time.time() - start_time
    print 'Done in %d seconds' % (end_time, )

if __name__ == '__main__':
    main()
