#!/usr/bin/env python
import numpy as np
import pandas as pd
import click as ck

from utils import (
    get_gene_ontology, get_anchestors, get_go_set,
    EXP_CODES,
    BIOLOGICAL_PROCESS,
    MOLECULAR_FUNCTION,
    CELLULAR_COMPONENT)


DATA_ROOT = 'data/swissexp/'
@ck.command()
def main():
    go = get_gene_ontology('go.obo')
    df = pd.read_pickle(DATA_ROOT + 'swissprot_exp.pkl')
    funcs = get_go_set(go, CELLULAR_COMPONENT)
    go_set = set()
    print(df)
    for i, row in df.iterrows():
        for go_id in row['annots']:
            it = go_id.split('|')
            go_id = it[0]
            if go_id in funcs and it[1] in EXP_CODES:
                go_set |= get_anchestors(go, go_id)
    print(len(go_set))


if __name__ == '__main__':
    main()
