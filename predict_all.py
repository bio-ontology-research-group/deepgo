#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from keras.models import load_model
from subprocess import Popen, PIPE
import time
from aaindex import INVALID_ACIDS

MAXLEN = 1002
models = list()
funcs = ['cc', 'mf', 'bp']


@ck.command()
@ck.option('--in-file', '-i', help='Input FASTA file', required=True)
@ck.option('--chunk-size', '-cs', default=1000, help='Number of sequences to read at a time')
@ck.option('--out-file', '-o', default='results.tsv', help='Output result file')
@ck.option('--mapping-file', '-m', default='', help='Mapping file for embeddings database')
@ck.option('--threshold', '-t', default=0.3, help='Prediction threshold')
@ck.option('--batch-size', '-bs', default=1, help='Batch size for prediction model')
@ck.option('--include-long-seq', '-ils', is_flag=True, help='Include long sequences')
def main(in_file, chunk_size, out_file, mapping_file, threshold, batch_size, include_long_seq):
    prot_ids = None
    mapping = None
    if mapping_file != '':
        mapping = read_mapping(mapping_file)
        prot_ids = {}

    w = open(out_file, 'w')
    for ids, sequences in read_fasta(in_file, chunk_size, include_long_seq):
        if mapping is not None:
            prot_ids = {}
            for i, seq_id in enumerate(ids):
                if seq_id in mapping:
                    prot_ids[mapping[seq_id]] = i
        results = predict_functions(sequences, prot_ids, batch_size, threshold)
        for i in range(len(ids)):
            w.write(ids[i])
            for res in results[i]:
                w.write('\t' + res)
            w.write('\n')
    w.close()


def read_mapping(mapping_file):
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            mapping[it[0]] = it[1]
    return mapping

def is_ok(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def read_fasta(filename, chunk_size, include_long_seq):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    if is_ok(seq):
                        if include_long_seq:
                            seqs.append(seq)
                            info.append(inf)
                            if len(info) == chunk_size:
                                yield (info, seqs)
                                seqs = list()
                                info = list()
                        elif len(seq) <= MAXLEN:
                            seqs.append(seq)
                            info.append(inf)
                            if len(info) == chunk_size:
                                yield (info, seqs)
                                seqs = list()
                                info = list()
                        else:
                            print(('Ignoring sequence {} because its length > 1002'
                              .format(inf)))
                    else:
                        print(('Ignoring sequence {} because of ambigious AA'
                              .format(inf)))
                    
                    seq = ''
                inf = line[1:].split()[0]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    yield (info, seqs)

def get_data(sequences, prot_ids):
    n = len(sequences)
    data = np.zeros((n, 1000), dtype=np.float32)
    embeds = np.zeros((n, 256), dtype=np.float32)

    if prot_ids is None:
        p = Popen(['diamond', 'blastp', '-d', 'data/embeddings',
                   '--max-target-seqs', '1', '--min-score', '60',
                   '--outfmt', '6', 'qseqid', 'sseqid'], stdin=PIPE, stdout=PIPE)
        for i in range(n):
            p.stdin.write(bytes('>' + str(i) + '\n' + sequences[i] + '\n'), encoding='utf-8')
        p.stdin.close()

        prot_ids = {}
        if p.wait() == 0:
            for line in p.stdout:
                it = line.decode('utf-8').strip().split('\t')
                if len(it) == 2:
                    prot_ids[it[1]] = int(it[0])

    prots = embed_df[embed_df['accessions'].isin(list(prot_ids.keys()))]
    for i, row in prots.iterrows():
        embeds[prot_ids[row['accessions']], :] = row['embeddings']

    for i in range(len(sequences)):
        seq = sequences[i]
        for j in range(min(MAXLEN, len(seq)) - gram_len + 1):
            data[i, j] = vocab[seq[j: (j + gram_len)]]
    return [data, embeds]


def predict(data, model, model_name, functions, threshold, batch_size):
    n = data[0].shape[0]
    result = list()
    for i in range(n):
        result.append(list())
    predictions = model.predict(
        data, batch_size=batch_size, verbose=1)
    for i in range(n):
        pred = (predictions[i] >= threshold).astype('int32')
        for j in range(len(functions)):
            if pred[j] == 1:
                result[i].append(model_name + '_' + functions[j] + '|' + '%.2f' % predictions[i][j])
    return result


def init_models(conf=None, **kwargs):
    print('Init')
    global models
    ngram_df = pd.read_pickle('data/models/ngrams.pkl')
    global embed_df
    embed_df = pd.read_pickle('data/graph_new_embeddings.pkl')
    global vocab
    vocab = {}
    global gram_len
    for key, gram in enumerate(ngram_df['ngrams']):
        vocab[gram] = key + 1
        gram_len = len(ngram_df['ngrams'][0])
    print(('Gram length:', gram_len))
    print(('Vocabulary size:', len(vocab)))
    threshold = 0.3
    # sequences = ['MKKVLVINGPNLNLLGIREKNIYGSVSYEDVLKSISRKAQELGFEVEFFQSNHEGEIIDKIHRAYFEKVDAIIINPGAYTHYSYAIHDAIKAVNIPTIEVHISNIHAREEFRHKSVIAPACTGQISGFGIKSYIIALYALKEILD']
    # data = get_data(sequences)
    for onto in funcs:
        model = load_model('data/models/model_%s.h5' % onto)
        df = pd.read_pickle('data/models/%s.pkl' % onto)
        functions = df['functions']
        models.append((model, functions))
        print('Model %s initialized.' % onto)
        # result = predict(data, model, functions, threshold)
        # print result


def predict_functions(sequences, prot_ids, batch_size, threshold):
    if not models:
        init_models()
    print('Predictions started')
    start_time = time.time()
    data = get_data(sequences, prot_ids)
    result = list()
    n = len(sequences)
    for i in range(n):
        result.append([])
    for i in range(len(models)):
        model, functions = models[i]
        print('Running predictions for model %s' % funcs[i])
        res = predict(data, model, funcs[i], functions, threshold, batch_size)
        for j in range(n):
            result[j] += res[j]
    print(('Predictions time: {}'.format(time.time() - start_time)))
    return result


if __name__ == '__main__':
    main()
