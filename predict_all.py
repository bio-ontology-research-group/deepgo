import click as ck
import numpy as np
import pandas as pd
from keras.models import load_model
from subprocess import Popen, PIPE

models = list()
funcs = ['cc', 'mf', 'bp']


@ck.command()
@ck.option('--input', '-i', help='Input FASTA file')
@ck.option('--output', '-o', help='Output result file')
def main(in_file, out_file):
    ids, sequences = read_fasta(in_file)
    results = predict_functions(data)
    df = pd.DataFrame({'id': ids, 'predictions': results})
    df.to_csv(out_file, sep='\t')


def read_fasta(lines):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if seq != '':
                seqs.append(seq)
                info.append(inf)
                seq = ''
            inf = line[1:].split()[0]
        else:
            seq += line
    seqs.append(seq)
    info.append(inf)
    return info, seqs

def get_data(sequences):
    n = len(sequences)
    data = np.zeros((n, 1000), dtype=np.float32)
    embeds = np.zeros((n, 256), dtype=np.float32)
    
    p = Popen(['blastp', '-db', 'data/embeddings.fa',
               '-max_target_seqs', '1', '-num_threads', '128',
               '-outfmt', '6 qseqid sseqid'], stdin=PIPE, stdout=PIPE)
    for i in xrange(n):
        p.stdin.write('>' + str(i) + '\n' + sequences[i] + '\n')
    p.stdin.close()

    prot_ids = {}
    if p.wait() == 0:
        for line in p.stdout:
            it = line.strip().split('\t')
            prot_ids[it[1]] = int(it[0])
    prots = embed_df[embed_df['accessions'].isin(prot_ids.keys())]
    for i, row in prots.iterrows():
        embeds[prot_ids[row['accessions']], :] = row['embeddings']
        
    for i in xrange(len(sequences)):
        seq = sequences[i]
        for j in xrange(len(seq) - gram_len + 1):
            data[i, j] = vocab[seq[j: (j + gram_len)]]
    return [data, embeds]


def predict(data, model, functions, threshold, batch_size=1):
    n = data[0].shape[0]
    result = list()
    for i in xrange(n):
        result.append(list())
    predictions = model.predict(
        data, batch_size=batch_size)
    for i in xrange(n):
        pred = (predictions[i] >= threshold).astype('int32')
        for j in xrange(len(functions)):
            if pred[j] == 1:
                result[i].append(functions[j])
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
    print('Gram length:', gram_len)
    print('Vocabulary size:', len(vocab))
    threshold = 0.3
    # sequences = ['MKKVLVINGPNLNLLGIREKNIYGSVSYEDVLKSISRKAQELGFEVEFFQSNHEGEIIDKIHRAYFEKVDAIIINPGAYTHYSYAIHDAIKAVNIPTIEVHISNIHAREEFRHKSVIAPACTGQISGFGIKSYIIALYALKEILD']
    # data = get_data(sequences)
    for onto in funcs:
        model = load_model('data/models/model_%s.h5' % onto)
        df = pd.read_pickle('data/models/%s.pkl' % onto)
        functions = df['functions']
        models.append((model, functions))
        print 'Model %s initialized.' % onto
        # result = predict(data, model, functions, threshold)
        # print result


def predict_functions(sequences, threshold=0.3):
    if not models:
        init_models()
    data = get_data(sequences)
    result = list()
    n = len(sequences)
    for i in xrange(n):
        result.append([])
    for i in range(len(models)):
        model, functions = models[i]
        print 'Running predictions for model %s' % funcs[i]
        res = predict(data, model, functions, threshold)
        for j in xrange(n):
            result[j] += res[j]
    return result
