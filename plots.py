#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import click as ck
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import math
from utils import (
    get_gene_ontology,
    BIOLOGICAL_PROCESS,
    MOLECULAR_FUNCTION,
    CELLULAR_COMPONENT,
    get_ipro,
    EXP_CODES
)

DATA_ROOT = 'data/swiss/'


@ck.command()
@ck.option('--function', default='mf', help='mf, cc, or bp')
def main(function):
    global FUNCTION
    FUNCTION = function
    # x, y = get_data(function + '.res')
    # plot(x, y)
    ipro_table()
    # plot_sequence_stats()
    # table()

def read_fasta(filename):
    data = list()
    c = 0
    with open(filename, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    data.append(seq)
                line = line[1:].split()[0].split('|')
                line = line[1] + '\t' + line[2]
                seq = line + '\t'
            else:
                seq += line
        data.append(seq)
    return data

def plot_sequence_stats():
    df = pd.read_pickle('data/swissprot.pkl')
    index = list()
    for i, row in df.iterrows():
        ok = False
        for it in row['annots']:
            it = it.split('|')
            if it[1] in EXP_CODES:
                ok = True
        if ok:
            index.append(i)
    df = df.iloc[index]
    print(len(df))
    lens = list(map(len, df['sequences']))
    c = 0
    for i in lens:
        if i <= 1002:
            c += 1
    print(c)
    h = np.histogram(lens, bins=(
        0, 500, 1000, 1500, 2000, 40000))
    plt.bar(list(range(5)),
        h[0], width=1, facecolor='green')
    titles = ['<=500', '<=1000', '<=1500', '<=2000', '>2000']
    plt.xticks(np.arange(0.5, 5.5, 1), titles)
    plt.xlabel('Sequence length')
    plt.ylabel('Sequence number')
    plt.title(r'Sequence length distribution')
    plt.savefig('sequence-dist.eps')
    print(np.max(lens))

def table():
    bp = get_data('bp.res')
    mf = get_data('mf.res')
    cc = get_data('cc.res')
    bp_seq = get_data('bp-seq.res')
    mf_seq = get_data('mf-seq.res')
    cc_seq = get_data('cc-seq.res')
    go = get_gene_ontology('go.obo')
    gos = go[BIOLOGICAL_PROCESS]['children']
    res = list()
    for go_id in gos:
        if go_id in bp:
            res.append((
                go_id, go[go_id]['name'],
                bp[go_id][0], bp[go_id][1],
                bp_seq[go_id][0], bp_seq[go_id][1]))
    for row in sorted(res, key=lambda x: x[2], reverse=True):
        print('%s & %s & %f & %f & %f & %f \\\\' % row)
    gos = go[MOLECULAR_FUNCTION]['children']
    print()
    res = list()
    for go_id in gos:
        if go_id in mf:
            res.append((
                go_id, go[go_id]['name'],
                mf[go_id][0], mf[go_id][1],
                mf_seq[go_id][0], mf_seq[go_id][1]))
    for row in sorted(res, key=lambda x: x[2], reverse=True):
        print('%s & %s & %f & %f & %f & %f \\\\' % row)
    gos = go[CELLULAR_COMPONENT]['children']
    print()
    res = list()
    for go_id in gos:
        if go_id in cc:
            res.append((
                go_id, go[go_id]['name'],
                cc[go_id][0], cc[go_id][1],
                cc_seq[go_id][0], cc_seq[go_id][1]))
    for row in sorted(res, key=lambda x: x[2], reverse=True):
        print('%s & %s & %f & %f & %f & %f \\\\' % row)


def get_ipro_data(filename):
    res = dict()
    with open(DATA_ROOT + filename) as f:
        for line in f:
            it = line.strip().split('\t')
            ipro_id = it[0]
            support = int(it[1])
            f = float(it[2])
            p = float(it[3])
            r = float(it[4])
            res[ipro_id] = (support, f, p, r)
    return res


def ipro_table():
    ipro = get_ipro()
    cc = get_ipro_data('ipro_cc.res')
    mf = get_ipro_data('ipro_mf.res')
    bp = get_ipro_data('ipro_bp.res')
    inter = set(cc).intersection(set(mf)).intersection(set(bp))
    res = list()
    sup = 50
    for key in inter:
        if bp[key][0] >= sup and mf[key][0] >= sup and cc[key][0] >= sup:
            res.append((
                key, ipro[key]['name'],
                bp[key][1], bp[key][2], bp[key][3],
                mf[key][1], mf[key][2], mf[key][3],
                cc[key][1], cc[key][2], cc[key][3]))
    res = sorted(res, key=lambda x: x[2], reverse=True)
    for item in res:
        print('%s & %s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % item)


def get_data(function):
    res = dict()
    x = list()
    y = list()
    with open(DATA_ROOT + function) as f:
        for line in f:
            if not line.startswith('GO:'):
                continue
            it = line.strip().split(' ')
            res[it[0]] = (float(it[1]), float(it[5]))
            x.append(float(it[1]))
            y.append(int(it[4]))
    # data = sorted(zip(functions, x, prec, rec), key=lambda x: -x[1])
    # return data[:20]
    # return y, x
    return res


def plot(x, y):
    plt.figure()
    plt.errorbar(
        x, y,
        fmt='o')
    plt.xlabel('Number of positives')
    plt.ylabel('Fmax measure')
    plt.title('Function centric performance - ' + FUNCTION.upper() + ' Ontology')
    plt.legend(loc="lower right")
    plt.savefig(FUNCTION + '.eps')


if __name__ == '__main__':
    main()
