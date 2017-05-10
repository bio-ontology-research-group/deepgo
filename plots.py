#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import click as ck
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import math
from utils import (
    get_gene_ontology,
    BIOLOGICAL_PROCESS,
    MOLECULAR_FUNCTION,
    CELLULAR_COMPONENT,
    get_ipro,
)

DATA_ROOT = 'data/swissexp/'


@ck.command()
def main():
    # x, y = get_data('cc.res')
    # plot(x, y)
    ipro_table()


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
    res = sorted(res, key=lambda x: x[1], reverse=True)
    for item in res:
        print('%s & %s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\' % item)


def get_data(function):
    res = dict()
    with open(DATA_ROOT + function) as f:
        for line in f:
            if not line.startswith('GO:'):
                continue
            it = line.strip().split(' ')
            res[it[0]] = (float(it[1]), float(it[5]))
    # data = sorted(zip(functions, fscores, prec, rec), key=lambda x: -x[1])
    # return data[:20]
    return res


def plot(x, y):
    plt.figure()
    plt.errorbar(
        x, y,
        fmt='o')
    plt.xlabel('Number of positives')
    plt.ylabel('Fmax measure')
    plt.title('Function centric performance - CC Ontology')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
