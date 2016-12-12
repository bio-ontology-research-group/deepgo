#!/usr/bin/env python

"""
OMP_NUM_THREADS=64 THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python nn_hierarchical_network.py
"""

import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import (
    Dense, Dropout, Activation, Input,
    Flatten, Highway, merge, BatchNormalization)
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import (
    Convolution1D, MaxPooling1D)
from keras.optimizers import Adam, RMSprop, Adadelta
from sklearn.metrics import classification_report
from utils import (
    shuffle,
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    get_parents,
    BIOLOGICAL_PROCESS,
    MOLECULAR_FUNCTION,
    CELLULAR_COMPONENT,
    DataGenerator,
    get_node_name,
    FUNC_DICT)
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import sequence
from keras import backend as K
import sys
from aaindex import (
    AAINDEX)
from collections import deque
import time
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
sys.setrecursionlimit(100000)

DATA_ROOT = 'data/cafa3/'
MAXLEN = 1000
REPLEN = 384
FUNCTION = 'mf'
if len(sys.argv) > 1:
    FUNCTION = sys.argv[1]

GO_ID = FUNC_DICT[FUNCTION]
go = get_gene_ontology('go.obo')
ORG = ''

func_df = pd.read_pickle(DATA_ROOT + FUNCTION + ORG + '.pkl')
functions = func_df['functions'].values
func_set = set(functions)
all_functions = get_go_set(go, GO_ID)
logging.info(len(functions))
go_indexes = dict()
for ind, go_id in enumerate(functions):
    go_indexes[go_id] = ind


def load_data(split=0.7):
    df = pd.read_pickle(DATA_ROOT + 'data' + ORG + '-' + FUNCTION + '.pkl')
    n = len(df)
    index = np.arange(n)
    train_n = int(n * split)
    np.random.seed(seed=5)
    np.random.shuffle(index)
    train_df = df.loc[df.index[index[:train_n]]]
    test_df = df.loc[df.index[index[train_n:]]]
    return reformat_data(train_df, test_df)


def reformat_data(train_df, test_df, validation_split=0.8):
    train_n = int(validation_split * len(train_df['indexes']))
    train_data = train_df[:train_n]['indexes'].values
    train_labels = train_df[:train_n]['labels'].values
    val_data = train_df[train_n:]['indexes'].values
    val_labels = train_df[train_n:]['labels'].values
    test_data = test_df['indexes'].values
    test_labels = test_df['labels'].values
    train_data = sequence.pad_sequences(train_data, maxlen=MAXLEN)
    val_data = sequence.pad_sequences(val_data, maxlen=MAXLEN)
    test_data = sequence.pad_sequences(test_data, maxlen=MAXLEN)
    rep_train_data = train_df[:train_n]['rep'].values
    rep_length = len(rep_train_data[0])
    shape = rep_train_data.shape
    rep_train_data = np.hstack(rep_train_data).reshape(shape[0], rep_length)
    rep_val_data = train_df[train_n:]['rep'].values
    shape = rep_val_data.shape
    rep_val_data = np.hstack(rep_val_data).reshape(shape[0], rep_length)
    rep_test_data = test_df['rep'].values
    shape = rep_test_data.shape
    rep_test_data = np.hstack(rep_test_data).reshape(shape[0], rep_length)
    train_data = (train_data, rep_train_data)
    val_data = (val_data, rep_val_data)
    test_data = (test_data, rep_test_data, test_df['gos'].values)
    shape = train_labels.shape
    train_labels = np.hstack(train_labels).reshape(shape[0], len(functions))
    train_labels = train_labels.transpose()
    shape = val_labels.shape
    val_labels = np.hstack(val_labels).reshape(shape[0], len(functions))
    val_labels = val_labels.transpose()
    shape = test_labels.shape
    test_labels = np.hstack(test_labels).reshape(shape[0], len(functions))
    test_labels = test_labels.transpose()

    return (
        (train_labels, train_data),
        (val_labels, val_data),
        (test_labels, test_data))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def get_feature_model():
    embedding_dims = 20
    max_features = 21
    model = Sequential()
    model.add(Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN,
        dropout=0.2))
    model.add(Convolution1D(
        nb_filter=32,
        filter_length=20,
        border_mode='valid',
        activation='relu',
        subsample_length=1))
    model.add(MaxPooling1D(pool_length=10, stride=5))
    model.add(Flatten())
    return model


def get_function_node(go_id, parent_models, output_dim):
    ind = go_indexes[go_id]
    name = get_node_name(ind * 3)
    output_name = get_node_name(ind * 3 + 1)
    merge_name = get_node_name(ind * 3 + 2)
    if len(parent_models) == 1:
        dense = Dense(
            output_dim,
            activation='relu',
            name=name)(parent_models[0])
    else:
        merged_parent_models = merge(
            parent_models, mode='concat', name=merge_name)
        dense = Dense(
            output_dim,
            activation='relu',
            name=name)(merged_parent_models)
    output = Dense(1, activation='sigmoid', name=output_name)(dense)
    return dense, output

def f_score(labels, preds):
    preds = K.round(preds)
    tp = K.sum(labels * preds)
    fp = K.sum(preds) - tp
    fn = K.sum(labels) - tp
    p = tp / (tp + fp)
    r = tp / (tp + fp)
    return 2 * p * r / (p + r)


def model():
    # set parameters:
    batch_size = 512
    nb_epoch = 100
    output_dim = 256
    nb_classes = len(functions)
    start_time = time.time()
    logging.info("Loading Data")
    train, val, test = load_data()
    train_labels, train_data = train
    val_labels, val_data = val
    test_labels, test_data = test
    logging.info("Data loaded in %d sec" % (time.time() - start_time))
    logging.info("Building the model")
    inputs = Input(shape=(MAXLEN,), dtype='int32', name='input1')
    inputs2 = Input(shape=(REPLEN,), dtype='float32', name='input2')
    feature_model = get_feature_model()(inputs)
    merged = merge([feature_model, inputs2], mode='concat', name='merged')
    go[GO_ID]['model'] = BatchNormalization()(merged)
    q = deque()
    used = set()
    for go_id in go[GO_ID]['children']:
        if go_id in func_set:
            q.append((go_id, output_dim))
            used.add(go_id)
    min_dim = output_dim
    while len(q) > 0:
        go_id, dim = q.popleft()
        min_dim = min(min_dim, dim)
        parents = get_parents(go, go_id)
        parent_models = list()
        for p_id in parents:
            if (p_id == GO_ID or p_id in func_set) and 'model' in go[p_id]:
                parent_models.append(go[p_id]['model'])
        dense, output = get_function_node(go_id, parent_models, dim)
        go[go_id]['model'] = dense
        go[go_id]['output'] = output
        for ch_id in go[go_id]['children']:
            if ch_id in func_set and ch_id not in used:
                q.append((ch_id, dim))
                used.add(ch_id)
    logging.info('Min dim %d' % min_dim)
    output_models = [None] * nb_classes
    for i in range(len(functions)):
        output_models[i] = go[functions[i]]['output']

    model = Model(input=[inputs, inputs2], output=output_models)
    logging.info('Model built in %d sec' % (time.time() - start_time))
    logging.info('Saving the model')
    model_json = model.to_json()
    with open(DATA_ROOT + 'model_network_' + FUNCTION + ORG + '.json', 'w') as f:
        f.write(model_json)
    logging.info('Compiling the model')
    optimizer = RMSprop()
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy')

    model_path = DATA_ROOT + 'hierarchical_network_' + FUNCTION + ORG + '.hdf5'
    checkpointer = ModelCheckpoint(
        filepath=model_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    logging.info(
        'Compilation finished in %d sec' % (time.time() - start_time))
    logging.info('Starting training the model')

    train_generator = DataGenerator(batch_size, nb_classes)
    train_generator.fit(train_data, train_labels)
    val_generator = DataGenerator(batch_size, nb_classes)
    val_generator.fit(val_data, val_labels)

    model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_data[0]),
        nb_epoch=nb_epoch,
        validation_data=val_generator,
        nb_val_samples=len(val_data[0]),
        max_q_size=batch_size,
        callbacks=[checkpointer, earlystopper])

    logging.info('Loading weights')
    model.load_weights(model_path)
    output_test = []
    for i in range(len(functions)):
        output_test.append(np.array(test_labels[i]))
    score = model.evaluate([test_data[0], test_data[1]], output_test, batch_size=batch_size)
    predictions = model.predict(
        [test_data[0], test_data[1]], batch_size=batch_size, verbose=1)

    prot_res = list()
    for i in range(len(test_data[0])):
        prot_res.append({
            'tp': 0.0, 'fp': 0.0, 'fn': 0.0,
            'pred': list(), 'test': list(), 'gos': test_data[2][i]})
    for i in range(len(test_labels)):
        rpred = predictions[i].flatten()
        pred = np.round(rpred)
        test = test_labels[i]
        for j in range(len(pred)):
            prot_res[j]['pred'].append(rpred[j])
            prot_res[j]['test'].append(test[j])
            if pred[j] == 1 and test[j] == 1:
                prot_res[j]['tp'] += 1
            elif pred[j] == 1 and test[j] == 0:
                prot_res[j]['fp'] += 1
            elif pred[j] == 0 and test[j] == 1:
                prot_res[j]['fn'] += 1
        logging.info(functions[i])
        logging.info(classification_report(test, pred))
    fs = 0.0
    p = 0.0
    r = 0.0
    n = 0
    for prot in prot_res:
        pred = prot['pred']
        test = prot['test']
        tp = prot['tp']
        fp = prot['fp']
        fn = prot['fn']
        for go_id in prot['gos']:
            if go_id not in func_set and go_id in all_functions:
                fn += 1
        if tp == 0.0 and fp == 0.0 and fn == 0.0:
            continue
        if tp != 0.0:
            recall = tp / (1.0 * (tp + fn))
            precision = tp / (1.0 * (tp + fp))
            p += precision
            r += recall
            fs += 2 * precision * recall / (precision + recall)
        n += 1
    logging.info('F measure: \t %f %f %f' % (fs / n, p / n, r / n))
    logging.info('Test loss:\t %f' % score[0])
    logging.info('Done in %d sec' % (time.time() - start_time))


def print_report(report, go_id):
    with open(DATA_ROOT + 'reports.txt', 'a') as f:
        f.write('Classification report for ' + go_id + '\n')
        f.write(report + '\n')


def main(*args, **kwargs):
    model()

if __name__ == '__main__':
    main(*sys.argv)
