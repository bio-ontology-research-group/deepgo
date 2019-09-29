#!/usr/bin/env python



import sys
import os
import pandas as pd
import numpy as np

DATA_ROOT = 'data/text/'
GENE_IRI = 'http://www.ncbi.nlm.nih.gov/gene/'


def run(*args):
    fw = open(DATA_ROOT + 'gene_to_rep.tab', 'w')
    with open(DATA_ROOT + 'gene_embedding.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            gene_id = items[0][len(GENE_IRI):].split('(')[0]
            if gene_id.find(';') == -1:
                fw.write(gene_id)
                for item in items[1:]:
                    fw.write('\t' + item)
                fw.write('\n')
    fw.close()


def ncbi2uni():
    res = dict()
    with open('data/uni2ncbi.tab') as f:
        for line in f:
            items = line.strip().split('\t')
            gene_id = items[1]
            res[gene_id] = items[0]
    with open('data/uniprot-gene-id.tab') as f:
        for line in f:
            items = line.strip().split('\t')
            if len(items) > 2:
                gene_ids = items[2].split(';')
                if gene_ids and gene_ids[0]:
                    for gene_id in gene_ids:
                        res[gene_id] = items[0]
    return res


def get_text_reps():
    text_reps = dict()
    uni_ids = ncbi2uni()
    with open(DATA_ROOT + 'gene_to_rep.tab') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[0] in uni_ids:
                text_reps[uni_ids[items[0]]] = np.array(
                    list(map(float, items[1:])), dtype='float32')
    return text_reps


def main(*args):
    run(*args)


if __name__ == '__main__':
    main(*sys.argv)
