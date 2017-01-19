#!/usr/bin/env python
from __future__ import print_function
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

DATA_ROOT = 'data/'


@ck.command()
def main():



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


if __name__ == '__main__':
    main()
