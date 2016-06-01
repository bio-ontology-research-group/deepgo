#!/usr/bin/env python

"""
KERAS_BACKEND=tensorflow python nn_hierarchical_tf.py
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
    DataGenerator)
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import sequence
import sys
from aaindex import (
    AAINDEX)
from collections import deque
import time
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
sys.setrecursionlimit(100000)

DATA_ROOT = 'data/swiss/'
MAXLEN = 1000
GO_ID = MOLECULAR_FUNCTION
go = get_gene_ontology('go.obo')


func_df = pd.read_pickle(DATA_ROOT + 'mf.pkl')
functions = func_df['functions'].values
func_set = set(functions)
logging.info(len(functions))
go_indexes = dict()
for ind, go_id in enumerate(functions):
    go_indexes[go_id] = ind


def load_data(validation_split=0.8):
    train_df = pd.read_pickle(DATA_ROOT + 'train-mf.pkl')
    test_df = pd.read_pickle(DATA_ROOT + 'test-mf.pkl')
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
        nb_filter=20,
        filter_length=10,
        border_mode='valid',
        activation='relu',
        subsample_length=1))
    model.add(MaxPooling1D(pool_length=5, stride=5))
    model.add(Flatten())
    return model


def get_function_node(go_id, parent_models, output_dim):
    if len(parent_models) == 1:
        dense = Dense(output_dim, activation='relu')(parent_models[0])
    else:
        merged_parent_models = merge(parent_models, mode='concat')
        dense = Dense(output_dim, activation='relu')(merged_parent_models)
    output = Dense(1, activation='sigmoid')(dense)
    return dense, output


def model():
    # set parameters:
    batch_size = 512
    nb_epoch = 100
    output_dim = 128
    nb_classes = len(functions)
    start_time = time.time()
    logging.info("Loading Data")
    train, val, test = load_data()
    train_labels, train_data = train
    val_labels, val_data = val
    test_labels, test_data = test
    logging.info("Data loaded in %d sec" % (time.time() - start_time))
    logging.info("Building the model")
    inputs = Input(shape=(MAXLEN,), dtype='int32')
    feature_model = get_feature_model()(inputs)
    go[GO_ID]['model'] = BatchNormalization()(feature_model)
    q = deque()
    for go_id in go[GO_ID]['children']:
        q.append((go_id, output_dim))
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
            if ch_id in func_set and 'model' not in go[ch_id]:
                q.append((ch_id, dim))
    logging.info('Min dim %d' % min_dim)
    output_models = [None] * nb_classes
    for i in range(len(functions)):
        output_models[i] = go[functions[i]]['output']

    model = Model(input=inputs, output=output_models)
    logging.info('Model built in %d sec' % (time.time() - start_time))
    logging.info('Compiling the model')
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    model_path = DATA_ROOT + 'hierarchical_mf.hdf5'
    checkpointer = ModelCheckpoint(
        filepath=model_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    logging.info(
        'Compilation finished in %d sec' % (time.time() - start_time))
    logging.info('Starting training the model')

    train_generator = DataGenerator(batch_size, nb_classes)
    train_generator.fit(train_data, train_labels)
    val_generator = DataGenerator(batch_size, nb_classes)
    val_generator.fit(val_data, val_labels)

    model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_data),
        nb_epoch=nb_epoch,
        validation_data=val_generator,
        nb_val_samples=len(val_data),
        max_q_size=batch_size,
        callbacks=[checkpointer, earlystopper])

    logging.info('Loading weights')
    model.load_weights(model_path)
    output_test = []
    for i in range(len(functions)):
        output_test.append(np.array(test_labels[i]))
    score = model.evaluate(test_data, output_test, batch_size=batch_size)
    predictions = model.predict(
        test_data, batch_size=batch_size, verbose=1)

    prot_res = list()
    for i in range(len(test_data)):
        prot_res.append({
            'tp': 0.0, 'fp': 0.0, 'fn': 0.0,
            'pred': list(), 'test': list()})
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
    n = 0
    with open(DATA_ROOT + 'predictions-mf.txt', 'w') as f:
        for prot in prot_res:
            pred = prot['pred']
            test = prot['test']
            f.write(str(pred[0]))
            for v in pred[1:]:
                f.write('\t' + str(v))
            f.write('\n')

            f.write(str(test[0]))
            for v in test[1:]:
                f.write('\t' + str(v))
            f.write('\n')

            tp = prot['tp']
            fp = prot['fp']
            fn = prot['fn']
            if tp == 0.0 and fp == 0.0 and fn == 0.0:
                continue
            if tp != 0.0:
                recall = tp / (1.0 * (tp + fn))
                precision = tp / (1.0 * (tp + fp))
                fs += 2 * precision * recall / (precision + recall)
            n += 1
    logging.info('Protein centric F measure: \t %f %d' % (fs / n, n))
    logging.info('Test loss:\t %f' % score[0])
    logging.info('Test accuracy:\t %f' % score[1])
    logging.info('Done in %d sec' % (time.time() - start_time))


def print_report(report, go_id):
    with open(DATA_ROOT + 'reports.txt', 'a') as f:
        f.write('Classification report for ' + go_id + '\n')
        f.write(report + '\n')


def main(*args, **kwargs):
    model()

if __name__ == '__main__':
    main(*sys.argv)
