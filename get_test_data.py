#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    BIOLOGICAL_PROCESS,
    MOLECULAR_FUNCTION,
    CELLULAR_COMPONENT)

DATA_ROOT = 'data/swiss/'
ORG = '-rat'


def get_org_proteins():
    proteins = set()
    with open(DATA_ROOT + 'proteins' + ORG + '.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            proteins.add(items[0])
    return proteins


def filter_org_test():
    proteins = get_org_proteins()
    fl = open(DATA_ROOT + 'test' + ORG + '.txt', 'w')
    with open(DATA_ROOT + 'test.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[0] in proteins:
                fl.write(line)
    fl.close()


def main(*args, **kwargs):
    filter_org_test()


if __name__ == '__main__':
    main(*sys.argv)
