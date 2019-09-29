#!/usr/bin/env python



import click as ck
import numpy as np
import pandas as pd
from sklearn import cluster


DATA_ROOT = 'data/clusters/'

@ck.command()
@ck.option('--function',
           default='mf',
           help='Function (mf, bp, cc)')
@ck.option('--threshold',
           default=0.5,
           help='Threshold for including protein to cluster')
def main(function, threshold):
    global FUNCTION
    FUNCTION = function
    data, prots = load_data(threshold)
    print('Proteins:', len(prots))
    df = pd.DataFrame({'proteins': prots})
    df.to_pickle(DATA_ROOT + 'clusters.pkl')


def load_data(threshold):
    sim = {}
    with open(DATA_ROOT + 'swiss.blst') as f:
        next(f)
        for line in f:
            it = line.strip().split('\t')
            prot1 = it[0]
            prot2 = it[1]
            if prot1 == prot2:
                continue
            score = float(it[2]) / 100.0
            if score < threshold:
                continue
            if prot1 not in sim:
                sim[prot1] = {}
            sim[prot1][prot2] = score
            if prot2 not in sim:
                sim[prot2] = {}
            sim[prot2][prot1] = score
    prots = list(sim.keys())
    n = len(prots)
    X = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if prots[j] in sim[prots[i]]:
                score = sim[prots[i]][prots[j]]
                X[i, j] = score
                X[j, i] = score
    return (X, prots)

if __name__ == '__main__':
    main()
