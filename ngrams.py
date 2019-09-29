#!/usr/bin/env python


import click as ck
import pandas as pd
from aaindex import is_ok


@ck.command()
@ck.option(
    '--length',
    default=3,
    help='Ngram length')
def main(length):
    seqs = get_sequences()
    ngrams = set()
    for seq in seqs:
        for i in range(len(seq) - length + 1):
            ngrams.add(seq[i: (i + length)])
    ngrams = list(sorted(ngrams))
    print(ngrams[:100])
    print(len(ngrams))
    df = pd.DataFrame({'ngrams': ngrams})
    df.to_pickle('data/cafa3/ngrams.pkl')


def get_sequences():
    data = list()
    with open('data/cafa3/targets.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if is_ok(items[1]):
                data.append(items[1])

    with open('data/cafa3/data.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            if is_ok(items[1]):
                data.append(items[1])
    return data


if __name__ == '__main__':
    main()
