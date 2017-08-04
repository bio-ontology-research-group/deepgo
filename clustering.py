#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import click as ck
import numpy as np
import pandas as pd
from sklearn import cluster


DATA_ROOT = 'data/blast/'

@ck.command()
@ck.option('--function',
           default='mf',
           help='Function (mf, bp, cc)')
def main(function):
    global FUNCTION
    FUNCTION = function
    data, prots = load_data()
    cls = cluster.AgglomerativeClustering(
        n_clusters=100, affinity='precomputed', linkage='complete')
    results = cls.fit_predict(data)
    print(results)

def load_data():
    sim = {}
    with open(DATA_ROOT + FUNCTION + '.blst') as f:
        for line in f:
            it = line.strip().split('\t')
            prot1 = it[0]
            prot2 = it[1]
            score = int(it[2])
            if prot1 not in sim:
                sim[prot1] = {}
            sim[prot1][prot2] = score
            if prot2 not in sim:
                sim[prot2] = {}
            sim[prot2][prot1] = score
    prots = sim.keys()
    n = len(prots)
    X = np.ndarray((n, n), dtype=np.float32)
    for i in xrange(n):
        for j in xrange(i + 1, n):
            if prots[j] in sim[prots[i]]:
                score = sim[prots[i]][prots[j]]
            else:
                score = 0.0
            X[i, j] = score
            X[j, i] = score
    mx = np.max(X)
    X = X / mx
    for i in xrange(n):
        X[i, i] = 1.0
    X = 1 - X
    print(X.shape, X)
    return (X, prots)

if __name__ == '__main__':
    main()
