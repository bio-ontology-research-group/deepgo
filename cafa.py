#!/usr/bin/env python
import sys
import os
import numpy as np
import pandas as pd
from aaindex import INVALID_ACIDS

MAXLEN = 1000


def is_ok(seq):
    if len(seq) > MAXLEN:
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
                line = line[1:]
                seq = line + '\t'
            else:
                seq += line
        data.append(seq)
    return data


def get_annotations():
    w = open('data/cafa2/annotations_2014.tab', 'w')
    with open('data/uniprot_sprot_2014.dat', 'r') as f:
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
    cafa_root = 'data/cafa3/CAFA3_targets/'
    data = list()
    # for dr in os.listdir(cafa_root):
    # if os.path.isdir(cafa_root + 'Targets/'):
    for fl in os.listdir(cafa_root + 'Targets/'):
        if fl.endswith('.fasta'):
            seqs = read_fasta(cafa_root + 'Targets/' + fl)
            data += seqs
    with open('data/cafa3/targets.txt', 'w') as f:
        for line in data:
            f.write(line + '\n')


def sprot2tabs():
    data = read_fasta('data/uniprot_sprot.fasta')
    with open('data/cafa2/uniprot_sprot.tab', 'w') as f:
        for line in data:
            f.write(line + '\n')


def cafa3():
    root = 'data/cafa3/CAFA3_training_data/'
    filename = root + 'uniprot_sprot_exp.fasta'
    data = read_fasta(filename)
    annots = dict()
    with open(root + 'uniprot_sprot_exp.txt') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[0] not in annots:
                annots[items[0]] = set()
            annots[items[0]].add(items[1])
    fl = open(root + 'uniprot_sprot.tab', 'w')
    for line in data:
        items = line.split('\t')
        if is_ok(items[1]) and items[0] in annots:
            fl.write(line + '\t')
            gos = list(annots[items[0]])
            fl.write(gos[0])
            for go_id in gos[1:]:
                fl.write('; ' + go_id)
            fl.write('\n')


def get_data():
    # targets = set()
    # with open('data/cafa2/targets.txt', 'r') as f:
    #     for line in f:
    #         items = line.strip().split()
    #         targets.add(items[1])
    seqs = dict()
    with open('data/cafa3/targets.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if is_ok(items[1]):
                prot_id = items[0]
                seqs[prot_id] = items[1]
    # print len(seqs)
    annots = dict()
    with open('data/cafa3/uniprot-go.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[1] in seqs:
                annots[items[1]] = items[2]

    # np.random.shuffle(data)
    fl = open('data/cafa3/data.txt', 'w')
    for prot_id in seqs:
        if prot_id in annots:
            fl.write(prot_id + '\t' + seqs[prot_id] + '\t' + annots[prot_id])
            fl.write('\n')
        else:
            print(prot_id)
    fl.close()


def cafa2string():
    rep_prots = set()
    with open('data/uni_mapping.tab') as f:
        for line in f:
            items = line.strip().split('\t')
            rep_prots.add(items[0])
    c = 0
    with open('data/cafa3/targets.txt') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[0] in rep_prots:
                c += 1
    print(c)


def main(*args, **kwargs):
    # get_data()
    # cafa3()
    fasta2tabs()
    # cafa2string()


if __name__ == '__main__':
    main(*sys.argv)
