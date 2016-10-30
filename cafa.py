#!/usr/bin/env python
import sys
import os
import numpy as np
import pandas as pd
from aaindex import INVALID_ACIDS

MINLEN = 25
MAXLEN = 1000


def is_ok(seq):
    if len(seq) < MINLEN or len(seq) > MAXLEN:
        return False
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def read_fasta(filename):
    data = list()
    with open(filename, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    data.append(seq)
                line = line.split()[0]
                seq = line + '\t'
            else:
                seq += line
        data.append(seq)
    return data


def get_annotations():
    w = open('data/cafa2/annotations.tab', 'w')
    with open('data/uniprot_sprot.dat', 'r') as f:
        prot = ''
        annots = list()
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot != '' and len(annots) > 0:
                    w.write(prot)
                    for go_id in annots:
                        w.write('\t' + go_id)
                    w.write('\n')
                prot = items[1]

                annots = list()
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    annots.append(items[1])
        if len(annots) > 0:
            w.write(prot)
            for go_id in annots:
                w.write('\t' + go_id)
            w.write('\n')
        w.close()


def fasta2tabs():
    cafa_root = 'data/cafa2/data/CAFA2-targets/'
    data = list()
    for dr in os.listdir(cafa_root):
        if os.path.isdir(cafa_root + dr):
            for fl in os.listdir(cafa_root + dr):
                if fl.endswith('.tfa'):
                    seqs = read_fasta(cafa_root + dr + '/' + fl)
                    data += seqs
    with open('data/cafa2/targets.txt', 'w') as f:
        for line in data:
            f.write(line + '\n')


def sprot2tabs():
    data = read_fasta('data/uniprot_sprot.fasta')
    with open('data/cafa2/uniprot_sprot.tab', 'w') as f:
        for line in data:
            f.write(line + '\n')


def get_data():
    targets = set()
    with open('data/cafa2/targets.txt', 'r') as f:
        for line in f:
            items = line.strip().split()
            targets.add(items[1])
    seqs = dict()
    with open('data/cafa2/uniprot_sprot.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if is_ok(items[1]):
                prot_id = items[0].split('|')[2]
                seqs[prot_id] = items[1]

    data = list()
    with open('data/cafa2/annotations-2013.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[0] not in targets and items[0] in seqs:
                data.append(items)

    np.random.shuffle(data)
    fl = open('data/cafa2/train.txt', 'w')
    for items in data:
        fl.write(items[0] + '\t' + seqs[items[0]] + '\t' + items[1])
        for i in range(2, len(items)):
            fl.write('; ' + items[i])
        fl.write('\n')
    fl.close()


def main(*args, **kwargs):
    get_data()


if __name__ == '__main__':
    main(*sys.argv)
