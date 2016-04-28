#!/usr/bin/env python
# Filename: get_sequence_function_data.py
import sys
from collections import deque
import time
from utils import get_gene_ontology
from matplotlib import (
    pyplot as plt,
    use as matplotlib_use)
matplotlib_use('Agg')


DATA_ROOT = 'data/'
FILES = (
    'uniprot-swiss.txt',)
INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X'])


go = get_gene_ontology('goslim_yeast.obo')


def get_go_set(go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set


functions = get_go_set('GO:0003674')


def isOk(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def main():
    start_time = time.time()
    go_set = get_go_set('GO:0003674')
    for g_id in go_set:
        print g_id
    print 'Starting filtering proteins with set of %d GOs' % (len(go_set),)
    min_len = sys.maxint
    max_len = 0
    lengths = list()
    for i in range(len(FILES)):
        file_name = FILES[i]
        with open(DATA_ROOT + file_name, 'r') as f:
            with open(DATA_ROOT + 'uniprot-swiss-mol-func-yeast.txt', 'w') as fall:
                for line in f:
                    line = line.strip().split('\t')
                    prot_id = line[0]
                    seq = line[1]
                    seq_len = len(seq)
                    if seq_len > 5000:
                        continue
                    lengths.append(seq_len)
                    gos = line[2]
                    if seq_len > 24 and isOk(seq):
                        go_ok = False
                        for g_id in gos.split('; '):
                            if g_id in functions:
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
    plt.plot(lengths)
    plt.show()

if __name__ == '__main__':
    main()
