#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    MOLECULAR_FUNCTION,
    BIOLOGICAL_PROCESS,
    CELLULAR_COMPONENT)
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

DATA_ROOT = 'data/'

GO_ID = BIOLOGICAL_PROCESS
ORG = 'yeast'

# go = get_gene_ontology('go.obo')

# func_df = pd.read_pickle(DATA_ROOT + 'bp.pkl')
# functions = func_df['functions'].values
# func_set = set(functions)


def filter_interactions():
    with open(DATA_ROOT + 'protein.links.v10.txt', 'r') as f:
        with open(DATA_ROOT + 'interactions.txt', 'w') as w:
            next(f)
            for line in f:
                items = line.strip().split(' ')
                score = float(items[2])
                if score >= 0.7:
                    w.write(line)


def get_interactions():
    gene_gene = dict()
    with open('data/interactions.' + ORG + '.tab', 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            items = line.strip().split('\t')
            gene1_id = list()
            gene1_id.append(items[0].upper())
            gene1_id.append(items[2].upper())
            gene2_id = list()
            gene2_id.append(items[1].upper())
            gene2_id.append(items[3].upper())
            for syn1 in items[4].split('|'):
                gene1_id.append(syn1)
            for syn2 in items[5].split('|'):
                gene2_id.append(syn2)
            for g1 in gene1_id:
                for g2 in gene2_id:
                    if g1 not in gene_gene:
                        gene_gene[g1] = set()
                    if g2 not in gene_gene:
                        gene_gene[g2] = set()
                    gene_gene[g1].add(g2)
                    gene_gene[g2].add(g1)
    return gene_gene


def get_interactions2():
    gene_gene = dict()
    with open('data/SGA_NxN.txt', 'r') as f:
        next(f)
        for line in f:
            items = line.strip().split('\t')
            g1 = items[1].upper()
            g2 = items[3].upper()
            score = float(items[5])
            if score >= 0.1:
                if g1 not in gene_gene:
                    gene_gene[g1] = set()
                if g2 not in gene_gene:
                    gene_gene[g2] = set()
                gene_gene[g1].add(g2)
                gene_gene[g2].add(g1)
    return gene_gene


def uni2gene():
    names = dict()
    with open('data/uni2gn.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if len(items) == 2:
                names[items[0].upper()] = items[1].split()[0].upper()
    return names


def uni2ensg():
    names = dict()
    with open('data/uni2ensg.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if len(items) == 2:
                names[items[0].upper()] = items[1].upper()
    return names


def load_proteins():
    proteins = list()
    with open('data/swiss/test-' + ORG + '-preds.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if len(items) == 1:
                continue
            proteins.append(items[0].upper())
    return proteins


def load_scores():
    scores = list()
    with open('data/similarities/' + ORG + '_resnik.txt', 'r') as f:
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


def compute_roc(scores, test):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # for i in range(n):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test, scores)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(roc_auc["micro"])
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(
       fpr["micro"],
       tpr["micro"],
       label='ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve BMA Resnik - ' + ORG)
    plt.legend(loc="lower right")
    plt.show()


def main(*args, **kwargs):
    gene_names = uni2gene()
    gene_ensg = uni2ensg()
    proteins = load_proteins()
    genes = list()
    for prot_id in proteins:
        if prot_id in gene_names:
            genes.append(gene_names[prot_id])
        elif prot_id in gene_ensg:
            genes.append(gene_ensg[prot_id])
        else:
            genes.append(prot_id)
    gene_gene = get_interactions2()
    # print(genes)
    inters = list()
    n = len(genes)
    for i in range(n):
        if genes[i] in gene_gene:
            gene_set = gene_gene[genes[i]]
            for j in range(n):
                if genes[j] in gene_set:
                    inters.append(1)
                else:
                    inters.append(0)
        else:
            for j in range(n):
                inters.append(0)

    inters = np.array(inters, dtype='int32')
    scores = load_scores()
    mn = np.min(scores)
    mx = np.max(scores)
    # Normalize scores
    scores = (scores - mn) / (mx - mn)
    assert len(scores) == len(inters)

    print(sum(inters), len(inters))

    compute_roc(scores, inters)


if __name__ == '__main__':
    main(*sys.argv)
