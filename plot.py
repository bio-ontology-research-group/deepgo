#!/usr/bin/env python
import os
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
FUNCTION = 'cc'


def load_ipro():
    support = list()
    fm = list()
    with open('data/per-' + FUNCTION + '.out', 'r') as f:
        for line in f:
            if not line.startswith('IPR'):
                continue
            line = line.strip().split(': ')[-1].strip().split(' ')
            if int(line[1]) > 500:
                continue
            fm.append(float(line[0]))
            support.append(int(line[1]))
    return fm, support


def load_go():
    support = list()
    fm = list()
    with open('data/per-' + FUNCTION + '.out', 'r') as f:
        for line in f:
            if line.startswith('GO'):
                next(f)
                next(f)
                next(f)
                p = next(f).split()
                print p
                fm.append(float(p[3]))
                support.append(int(p[4]))
    return fm, support

def draw():
    fm, support = load_go()
    plt.errorbar(support, fm, fmt='o', label='title')
    plt.legend()
    plt.xlabel('Support')
    plt.ylabel('F measure')
    plt.title('Title')

    plt.savefig('go-' + FUNCTION + '.pdf')


def main(*args, **kwargs):
    draw()

if __name__ == '__main__':
    main(*sys.argv)
