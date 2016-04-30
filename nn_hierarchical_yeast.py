#!/usr/bin/env python

"""
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rnn_sequence.py
"""

import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import (
    Dense, Dropout, Activation, Input, Flatten, Highway, merge)
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import (
    Convolution1D, MaxPooling1D)
from sklearn.metrics import classification_report
from utils import (
    shuffle,
    get_gene_ontology)
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import sequence
import sys
from aaindex import (
    AAINDEX)
from collections import deque

DATA_ROOT = 'data/yeast/'
MAXLEN = 1000
go = get_gene_ontology('goslim_yeast.obo')


def get_go_set(go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set

functions = get_go_set('GO:0003674')
functions.remove('GO:0003674')
functions = list(functions)
go_indexes = dict()
for ind, go_id in enumerate(functions):
    go_indexes[go_id] = ind


def load_data():
    train_df = pd.read_pickle(DATA_ROOT + 'train.pkl')
    test_df = pd.read_pickle(DATA_ROOT + 'test.pkl')

    train_data = train_df['indexes'].values
    train_labels = train_df['labels'].values
    test_data = test_df['indexes'].values
    test_labels = test_df['labels'].values
    train_data = sequence.pad_sequences(train_data, maxlen=MAXLEN)
    test_data = sequence.pad_sequences(test_data, maxlen=MAXLEN)
    shape = train_labels.shape
    train_labels = np.hstack(train_labels).reshape(shape[0], len(functions))
    train_labels = train_labels.transpose()
    shape = test_labels.shape
    test_labels = np.hstack(test_labels).reshape(shape[0], len(functions))
    test_labels = test_labels.transpose()

    return (
        (train_labels, train_data),
        (test_labels, test_data))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def get_feature_model():
    embedding_dims = 20
    max_features = 20
    model = Sequential()
    model.add(Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN,
        dropout=0.2))
    model.add(Convolution1D(
        nb_filter=64,
        filter_length=8,
        border_mode='valid',
        activation='relu',
        subsample_length=1))
    model.add(MaxPooling1D(pool_length=4))
    model.add(Flatten())
    return model


def get_function_node(go_id, parent_models):
    if len(parent_models) == 1:
        dense = Dense(128, activation='relu')(parent_models[0])
    else:
        merged_parent_models = merge(parent_models, mode='concat')
        dense = Dense(128, activation='relu')(merged_parent_models)
    output = Dense(1, activation='sigmoid', name=go_id)(dense)
    return dense, output


def model():
    # set parameters:
    batch_size = 256
    nb_epoch = 100
    nb_classes = len(functions)
    print "Loading Data"
    train, test = load_data()
    train_labels, train_data = train
    test_labels, test_data = test
    print "Building the model"
    inputs = Input(shape=(MAXLEN,), dtype='int32', name='input')
    feature_model = get_feature_model()(inputs)
    go['GO:0003674']['model'] = feature_model
    q = deque()
    for go_id in go['GO:0003674']['children']:
        q.append(go_id)

    while len(q) > 0:
        go_id = q.popleft()
        parent_models = list()
        for p_id in go[go_id]['is_a']:
            parent_models.append(go[p_id]['model'])
        dense, output = get_function_node(go_id, parent_models)
        go[go_id]['model'] = dense
        go[go_id]['output'] = output
        for ch_id in go[go_id]['children']:
            q.append(ch_id)

    output_models = [None] * nb_classes
    for go_id, ind in go_indexes.iteritems():
        output_models[ind] = go[go_id]['output']

    model = Model(input=inputs, output=output_models)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model_path = DATA_ROOT + 'hierarchical.hdf5'
    checkpointer = ModelCheckpoint(
        filepath=model_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    output = {}
    for i in range(len(functions)):
        output[functions[i]] = np.array(train_labels[i])
    model.fit(
        {'input': train_data},
        output,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        validation_split=0.2,
        callbacks=[checkpointer, earlystopper])

    print 'Loading weights'
    model.load_weights(model_path)
    output_test = {}
    for i in range(len(functions)):
        output_test[functions[i]] = np.array(test_labels[i])
    score = model.evaluate(
        {'input': test_data}, output_test, batch_size=batch_size)
    predictions = model.predict(
        test_data, batch_size=batch_size, verbose=1)

    prot_res = list()
    for i in range(len(test_data)):
        prot_res.append({'tp': 0.0, 'fp': 0.0, 'fn': 0.0})
    for i in range(len(test_labels)):
        pred = np.round(predictions[i].flatten())
        test = test_labels[i]
        for j in range(len(pred)):
            if pred[j] == 1 and test[j] == 1:
                prot_res[j]['tp'] += 1
            elif pred[j] == 1 and test[j] == 0:
                prot_res[j]['fp'] += 1
            elif pred[j] == 0 and test[j] == 1:
                prot_res[j]['fn'] += 1
        print functions[i]
        print classification_report(test, pred)
    f = 0.0
    n = 0
    for prot in prot_res:
        tp = prot['tp']
        fp = prot['fp']
        fn = prot['fn']
        if tp + fn > 0 and tp + fp > 0:
            recall = tp / (1.0 * (tp + fn))
            precision = tp / (1.0 * (tp + fp))
            if recall + precision != 0:
                f += 2 * precision * recall / (precision + recall)
            n += 1
    print 'Protein centric F measure: \t', f / n
    print 'Test loss:\t', score[0]
    print 'Test accuracy:\t', score[1]


def print_report(report, go_id):
    with open(DATA_ROOT + 'reports.txt', 'a') as f:
        f.write('Classification report for ' + go_id + '\n')
        f.write(report + '\n')


def main(*args, **kwargs):
    model()

if __name__ == '__main__':
    main(*sys.argv)
