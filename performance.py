#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd

DATA_ROOT = 'data/swiss/model-mf-1364/'


def compute_performance():
    fs = 0.0
    n = 0
    t = 0.7
    with open(DATA_ROOT + 'predictions.txt', 'r') as f:
        for line in f:
            preds = map(float, line.strip().split('\t'))
            tests = map(float, next(f).strip().split('\t'))
            preds = np.array(preds)
            for i in range(len(preds)):
                preds[i] = 0.0 if preds[i] < t else 1.0
            tests = np.array(tests)
            tp = 0.0
            fp = 0.0
            fn = 0.0
            for i in range(len(preds)):
                if tests[i] == 1 and preds[i] == 1:
                    tp += 1
                elif tests[i] == 1 and preds[i] == 0:
                    fn += 1
                elif tests[i] == 0 and preds[i] == 1:
                    fp += 1
            if tp == 0.0 and fp == 0.0 and fn == 0.0:
                continue
            if tp != 0.0:
                recall = tp / (1.0 * (tp + fn))
                precision = tp / (1.0 * (tp + fp))
                fs += 2 * precision * recall / (precision + recall)
            n += 1
    print 'Threshold:', t
    print 'Protein centric F-measure:', fs / n


def main(*args, **kwargs):
    compute_performance()

if __name__ == '__main__':
    main(*sys.argv)
