import click as ck
import numpy as np
import pandas as pd
from multiprocessing import Pool
from subprocess import Popen, PIPE, DEVNULL
import os

@ck.command()
def main():
    df = pd.read_pickle('data/latest/data.pkl')
    sequences = list()
    for i, row in df.iterrows():
        prot_id = row['proteins']
        seq = row['sequences']
        sequences.append((prot_id, seq))
    pool = Pool(16)
    pool.map(run_psiblast, sequences)

def run_psiblast(data):
    prot_id, seq = data
    org = prot_id.split('_')[1]
    dirname = 'data/pssm/' + org + '/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + prot_id + '.pssm'
    p = Popen([
        'psiblast', '-db', 'data/uniprot_sprot.fasta',
        '-num_iterations', '3', '-out_ascii_pssm', filename
    ], stdin=PIPE, stdout=DEVNULL, encoding='utf8')

    p.stdin.write('>' + prot_id + '\n' + seq + '\n')
    p.stdin.close()

    if p.wait() == 0:
        print('Success: ', prot_id)
    else:
        print('Failure: ', prot_id)

if __name__ == '__main__':
    main()
