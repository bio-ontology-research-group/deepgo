#!/usr/bin/env python
# Filename: get_sequence_function_data.py
import sys
from collections import deque
import time
from utils import (
    get_gene_ontology,
    get_go_set,
    BIOLOGICAL_PROCESS,
    MOLECULAR_FUNCTION,
    CELLULAR_COMPONENT)
from aaindex import INVALID_ACIDS
from matplotlib import (
    pyplot as plt,
    use as matplotlib_use)
matplotlib_use('Agg')


DATA_ROOT = 'data/'

go = get_gene_ontology('goslim_yeast.obo')

go_set = get_go_set(go, MOLECULAR_FUNCTION)


def isOk(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def filter_uniprot_go():
    with open(DATA_ROOT + 'uniprot-swiss.tab', 'r') as f:
        with open(DATA_ROOT + 'uniprot-swiss.txt', 'w') as fall:
            for line in f:
                items = line.strip().split('\t')
                if len(items) == 3:
                    fall.write(line)


def main():
    start_time = time.time()
    print 'Starting filtering proteins with set of %d GOs' % (len(go_set),)
    min_len = sys.maxint
    max_len = 0
    lengths = list()
    file_name = 'uniprot-swiss.txt'
    with open(DATA_ROOT + file_name, 'r') as f:
        with open(DATA_ROOT + 'uniprot-swiss-yeast-mf.txt', 'w') as fall:
            for line in f:
                line = line.strip().split('\t')
                prot_id = line[0]
                seq = line[1]
                seq_len = len(seq)
                lengths.append(seq_len)
                if seq_len > 1000:
                    continue
                gos = line[2]
                if seq_len > 24 and isOk(seq):
                    go_ok = False
                    for g_id in gos.split('; '):
                        if g_id in go_set:
                            go_ok = True
                            break
                    if go_ok:
                        fall.write(
                            prot_id + '\t' + seq + '\t' + gos + '\n')
                    min_len = min(min_len, seq_len)
                    max_len = max(max_len, seq_len)
    end_time = time.time() - start_time
    print 'Minimum length of sequences:', min_len
    print 'Maximum length of sequences:', max_len
    print 'Done in %d seconds' % (end_time, )
    # lengths = sorted(lengths)
    # plt.plot(lengths)
    # plt.show()

if __name__ == '__main__':
    main()
