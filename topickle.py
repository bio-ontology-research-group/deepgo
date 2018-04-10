#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
from utils import (
    get_gene_ontology,
    get_anchestors)
from aaindex import is_ok

MAXLEN = 2000
DATA_ROOT = 'data/latest/'

@ck.command()
def main():
    go = get_gene_ontology(DATA_ROOT + 'go.obo', with_rels=True)
    proteins = list()
    accessions = list()
    functions = list()
    with open(DATA_ROOT + 'annots.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            prot_id = it[0]
            acc_id = it[1].split(';')[0]
            gos = it[2:]
            go_set = set()
            for go_id in gos:
                go_id = go_id.split('|')[0]
                if go_id in go:
                    go_set |= get_anchestors(go, go_id)
            proteins.append(prot_id)
            accessions.append(acc_id)
            functions.append(go_set)
    
    sequences = list()
    with open(DATA_ROOT + 'sequences.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            sequences.append(it[1])
    df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'sequences': sequences,
        'functions': functions})
    df.to_pickle(DATA_ROOT + 'swissprot_exp.pkl')
    print(df)
        

if __name__ == '__main__':
    main()
