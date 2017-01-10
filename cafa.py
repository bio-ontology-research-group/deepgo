#!/usr/bin/env python
import sys
import os
import numpy as np
import pandas as pd
from aaindex import is_ok

MAXLEN = 1000


def get_fly_mapping():
    map1 = dict()
    with open('data/fly_uni.dat') as f:
        for line in f:
            it = line.strip().split('\t')
            map1[it[0]] = it[1]
    res = dict()
    with open('data/fly_idmapping.dat') as f:
        for line in f:
            it = line.strip().split('\t')
            if it[0] in map1:
                res[it[1]] = map1[it[0]]
    return res


def read_fasta(filename):
    data = list()
    org = os.path.basename(filename).split('.')[1]
    if org == '7227':
        mapping = get_fly_mapping()
    c = 0
    with open(filename, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    data.append(seq)
                line = line[1:].split()
                if org == '7227':
                    if line[1] in mapping:
                        line = org + '\t' + line[0] + '\t' + mapping[line[1]]
                    else:
                        line = org + '\t' + line[0] + '\t' + line[1]
                        c += 1
                else:
                    line = org + '\t' + line[0] + '\t' + line[1]
                seq = line + '\t'
            else:
                seq += line
        data.append(seq)
    print(c)
    return data


def get_annotations():
    w = open('data/cafa3/swissprot.tab', 'w')
    with open('data/uniprot_sprot.dat', 'r') as f:
        prot_id = ''
        prot_ac = ''
        annots = list()
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '' and len(annots) > 0:
                    w.write(prot_id + '\t' + prot_ac)
                    for go_id in annots:
                        w.write('\t' + go_id)
                    w.write('\n')
                prot_id = items[1]
                annots = list()
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac = items[1]
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    annots.append(go_id + '|' + code)

        if len(annots) > 0:
            w.write(prot_id + '\t' + prot_ac)
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
    with open('data/cafa3/uniprot_sprot.tab', 'w') as f:
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
    proteins = list()
    targets = list()
    orgs = list()
    ngrams = list()
    ngram_df = pd.read_pickle('data/cafa3/ngrams.pkl')
    vocab = {}
    for key, gram in enumerate(ngram_df['ngrams']):
        vocab[gram] = key + 1
    gram_len = len(ngram_df['ngrams'][0])
    print('Gram length:', gram_len)
    print('Vocabulary size:', len(vocab))

    with open('data/cafa3/targets.txt') as f:
        for line in f:
            it = line.strip().split('\t')
            seq = it[3]
            if is_ok(seq):
                orgs.append(it[0])
                targets.append(it[1])
                proteins.append(it[2])
                grams = np.zeros((len(seq) - gram_len + 1, ), dtype='int32')
                for i in xrange(len(seq) - gram_len + 1):
                    grams[i] = vocab[seq[i: (i + gram_len)]]
                ngrams.append(grams)

    df = pd.DataFrame({
        'targets': targets,
        'proteins': proteins,
        'ngrams': ngrams,
        'orgs': orgs})
    print(len(df))
    df.to_pickle('data/cafa3/targets.pkl')


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
    get_data()
    # cafa3()
    # fasta2tabs()
    # cafa2string()
    # get_annotations()
    # sprot2tabs()


if __name__ == '__main__':
    main(*sys.argv)
