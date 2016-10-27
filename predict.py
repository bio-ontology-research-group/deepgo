from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from keras.models import Model, model_from_json
from keras.preprocessing import sequence
from aaindex import AAINDEX
from utils import (
    get_gene_ontology,
    get_anchestors)

go = get_gene_ontology('go.obo')

models = list()
ontos = ['cc', 'mf', 'bp']
DATA_ROOT = 'data/swiss/'
MAXLEN = 1000
ORG = '-rat'


def get_data(sequences):
    n = len(sequences)
    data = np.zeros((n, MAXLEN), dtype='float32')
    for i in range(n):
        for j in range(len(sequences[i])):
            data[i, j] = AAINDEX[sequences[i][j]]
    return data


def predict(data, model, functions):
    batch_size = 1
    n = data.shape[0]
    result = list()
    for i in range(n):
        result.append(list())
    predictions = model.predict(
        data, batch_size=batch_size)
    for i in range(len(functions)):
        rpred = predictions[i].flatten()
        pred = np.round(rpred)
        for j in range(n):
            if pred[j] == 1:
                result[j].append(functions[i])
    return result


def init_models(org='', **kwargs):
    global models
    for func in ontos:
        with open(DATA_ROOT + 'model_' + func + org + '.json', 'r') as f:
            json_string = next(f)
        model = model_from_json(json_string)
        model.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])
        model.load_weights(DATA_ROOT + 'hierarchical_' + func + org + '.hdf5')
        df = pd.read_pickle(DATA_ROOT + func + org + '.pkl')
        functions = df['functions']
        models.append((model, functions))


def predict_functions(sequences):
    if not models:
        init_models(org=ORG)
    data = get_data(sequences)
    result = list()
    for i in range(len(models)):
        model, functions = models[i]
        print('Running predictions for model %s %s' % (ontos[i], ORG))
        result += predict(data, model, functions)
    return result


def filter_specific(gos):
    go_set = set()
    for go_id in gos:
        go_set.add(go_id)
    for go_id in gos:
        anchestors = get_anchestors(go, go_id)
        anchestors.discard(go_id)
        go_set -= anchestors
    return list(go_set)


def main(*args, **kwargs):
    prots = list()
    sequences = list()
    with open(DATA_ROOT + 'test' + ORG + '.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            prots.append(items[0])
            sequences.append(items[1])
    preds = predict_functions(sequences)
    n = len(prots)
    cc = preds[0: n]
    mf = preds[n: n + n]
    bp = preds[2 * n: 3 * n]

    with open(DATA_ROOT + 'test' + ORG + '-preds.txt', 'w') as f:
        for i in xrange(n):
            funcs = filter_specific(cc[i] + mf[i] + bp[i])
            f.write(prots[i])
            for func in funcs:
                f.write('\t' + func)
            f.write('\n')


if __name__ == '__main__':
    main(*sys.argv)
