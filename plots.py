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

DATA_ROOT = 'data/swissexp/'


@ck.command()
def main():
    x, y = get_data('cc.res')
    plot(x, y)


def table():
    bp = get_data('bp.res')
    mf = get_data('mf.res')
    cc = get_data('cc.res')
    for b, m, c in zip(bp, mf, cc):
        print(
            '%s & %.2f & %.2f & %.2f & %s & %.2f & %.2f & %.2f & %s & %.2f & %.2f & %.2f \\\\' % (b[0], b[1], b[2], b[3], m[0], m[1], m[2], m[3], c[0], c[1], c[2], c[3]))


def get_data(function):
    fscores = list()
    support = list()
    functions = list()
    prec = list()
    rec = list()
    with open(DATA_ROOT + function) as f:
        for line in f:
            if not line.startswith('GO:'):
                continue
            it = line.strip().split(' ')
            fs = float(it[1])
            sup = int(it[4])
            fscores.append(fs)
            prec.append(float(it[2]))
            rec.append(float(it[3]))
            support.append(sup)
            functions.append(it[0])
    # data = sorted(zip(functions, fscores, prec, rec), key=lambda x: -x[1])
    # return data[:20]
    return support, fscores


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
