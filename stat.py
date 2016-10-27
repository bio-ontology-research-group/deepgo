#!/usr/bin/env python

from __future__ import print_function
import sys
import pandas as pd

text = """Human - 0.735824 83726, 0.816713 86648, 0.791228 76303
Pombe - 0.764732 83717, 0.774527 86645, 0.789694 76126
Slime Mold - 0.753422 83722, 0.822216 86637, 0.802210 76110
Ecoli - 0.786364 83720, 0.842853 86643, 0.823757 76296
Mouse - 0.670736 83726, 0.833975 86644, 0.797558 76298
Rat - 0.763520 83726, 0.814845 86646, 0.759258 76296
Yeast - 0.775490 83693, 0.819911 86639, 0.792147 75995
Worm - 0.724621 83726, 0.837197 86639, 0.812929 75913
Zebrafish - 0.768210 83726, 0.819566 86639, 0.798909 76273
Fly - 0.759773 83726, 0.777102 86644, 0.827972 76314
Virus - 0.737761 83720, 0.822183 86640, 0.846400 76292"""


def main(*args):
    results = text.split('\n')
    for res in results:
        items = res.split(' - ')
        name = items[0]
        values = items[1].split(', ')
        print(
            name, '&',
            values[0].split()[0], '&',
            values[1].split()[0], '&',
            values[2].split()[0], '\\\\')


if __name__ == '__main__':
    main(*sys.argv[1:])
