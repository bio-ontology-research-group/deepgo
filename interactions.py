#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import click as ck
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    MOLECULAR_FUNCTION,
    BIOLOGICAL_PROCESS,
    CELLULAR_COMPONENT)
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import math

DATA_ROOT = 'data/'


@ck.command()
def main():
    scores = load_scores()
    m = int(math.sqrt(len(scores)))
    inters, index = load_interactions()
    n = len(index)
    print(n)
    proteins = list(index.keys())
    new_scores = list()
    interactions = list()
    for i in range(n):
        for j in range(n):
            x = index[proteins[i]]
            y = index[proteins[j]]
            new_scores.append(scores[m * x + y])
            interactions.append(inters[m * x + y])
    print(len(interactions))
    compute_roc(new_scores, interactions)


def get_data():
    df = pd.read_pickle('data/cafa3/swissprot_exp.pkl')
    training = set(df['proteins'])
    testing = set()
    with open('data/cafa3/CAFA3_targets/Targets/target.9606.fasta') as f:
        for line in f:
            if line.startswith('>'):
                it = line.strip().split()
                if it[1] not in training:
                    testing.add(it[1])
    with open('data/cafa3/human_test.tab', 'w') as f:
        for prot in testing:
            f.write(prot + '\n')

    with open('data/cafa3/human_train.tab', 'w') as f:
        for prot in training:
            if prot.endswith('_HUMAN'):
                f.write(prot + '\n')


def load_scores():
    scores = list()
    with open('data/cafa3/sim_merged.txt', 'r') as f:
        for line in f:
            line = line.strip()
            scores.append(float(line))
    res = np.array(scores, dtype='float32')
    n = res.shape[0]
    i = 0
    while i * i < n:
        res[i * i] = 0.0
        i += 1
    return res


def uni2string():
    df = pd.read_pickle('data/idmapping.9606.pkl')
    mapping = dict()
    for i, row in df.iterrows():
        if isinstance(row['string'], str):
            mapping[row['proteins']] = row['string']
    return mapping


def load_proteins():
    proteins = list()
    mapping = uni2string()
    with open('data/cafa3/test_merged_human.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            string_id = None
            if it[0] in mapping:
                string_id = mapping[it[0]]
            proteins.append((it[0], string_id))
    return proteins


def load_interactions():
    proteins = load_proteins()
    index = {}
    for i, prot in enumerate(proteins):
        prot_id, st_id = prot
        if st_id:
            index[st_id] = i
    n = len(proteins)
    inters = np.zeros((n * n, ), dtype='int32')
    with open('data/9606.protein.links.v10.txt') as f:
        next(f)
        for line in f:
            it = line.strip().split(' ')
            if int(it[2]) < 700:
                continue
            if it[0] in index and it[1] in index:
                x = index[it[0]]
                y = index[it[1]]
                inters[x * n + y] = 1
                inters[y * n + x] = 1
    return inters, index


def compute_roc(scores, test):
    fpr, tpr, _ = roc_curve(test, scores)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve BMA Resnik Human Merged')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
