#!/usr/bin/env python

"""
OMP_NUM_THREADS=64 THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python nn_hierarchical_network.py
"""

import numpy as np
import pandas as pd
import click as ck
from keras.models import Sequential, Model
from keras.layers import (
    Dense, Dropout, Activation, Input,
    Flatten, Highway, merge, BatchNormalization)
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import (
    Convolution1D, MaxPooling1D)
from keras.layers.recurrent import LSTM
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
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
sys.setrecursionlimit(100000)

DATA_ROOT = 'data/cafa3/'
MAXLEN = 1000
REPLEN = 384


@ck.command()
@ck.option(
    '--function',
    default='mf',
    help='Ontology id (mf, bp, cc)')
@ck.option(
    '--device',
    default='gpu:0',
    help='GPU or CPU device id')
@ck.option(
    '--org',
    default='',
    help='Organism name')
def main(function, device, org):
    global FUNCTION
    FUNCTION = function
    GO_ID = FUNC_DICT[FUNCTION]
    global go
    go = get_gene_ontology('go.obo')
    global ORG
    ORG = org
    func_df = pd.read_pickle(DATA_ROOT + FUNCTION + ORG + '.pkl')
    global functions
    functions = func_df['functions'].values
    global func_set
    func_set = set(functions)
    global all_functions
    all_functions = get_go_set(go, GO_ID)
    logging.info(len(functions))
    global go_indexes
    go_indexes = dict()
    for ind, go_id in enumerate(functions):
        go_indexes[go_id] = ind

    with tf.device('/' + device):
        model()


def load_data(split=0.7):
    df = pd.read_pickle(DATA_ROOT + 'data' + ORG + '-' + FUNCTION + '.pkl')
    n = len(df)
    index = np.arange(n)
    np.random.seed(5)
    np.random.shuffle(index)
    train_n = int(n * split)
    valid_n = int(train_n * 0.8)

    train_df = df.loc[index[:valid_n]]
    valid_df = df.loc[index[valid_n:train_n]]
    test_df = df.loc[index[train_n:]]

    def reshape(values):
        values = np.hstack(values).reshape(
            len(values), len(values[0]))
        return values

    def normalize_minmax(values):
        mn = np.min(values)
        mx = np.max(values)
        if mx - mn != 0.0:
            return (values - mn) / (mx - mn)
        return values - mn

    def get_values(data_frame):
        labels = reshape(data_frame['labels'].values)
        trigrams = sequence.pad_sequences(
            data_frame['trigrams'].values, maxlen=MAXLEN)
        trigrams = reshape(trigrams)
        rep = reshape(data_frame['rep'].values)
        data = (trigrams, rep)
        return data, labels

    train = get_values(train_df)
    valid = get_values(valid_df)
    test = get_values(test_df)
    gos = test_df['gos'].values

    return train, valid, test, gos


def get_feature_model():
    embedding_dims = 64
    max_features = 8001
    model = Sequential()
    model.add(Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN,
        dropout=0.2))
    # model.add(LSTM(128, activation='relu'))
    model.add(Convolution1D(
        nb_filter=32,
        filter_length=64,
        border_mode='valid',
        activation='relu',
        subsample_length=1))
    model.add(MaxPooling1D(pool_length=10, stride=5))
    model.add(Flatten())
    return model


def f_score(labels, preds):
    preds = K.round(preds)
    tp = K.sum(labels * preds)
    fp = K.sum(preds) - tp
    fn = K.sum(labels) - tp
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def merge_outputs(outputs, name):
    if len(outputs) == 1:
        return outputs[0]
    return merge(outputs, mode='concat', name=name, concat_axis=1)


def get_function_node(go_id, parent_models, output_dim):
    ind = go_indexes[go_id]
    name = get_node_name(ind * 3)
    output_name = get_node_name(ind * 3 + 1)
    merge_name = get_node_name(ind * 3 + 2)
    net = merge_outputs(parent_models, merge_name)
    net = Dense(output_dim, activation='relu', name=name)(net)
    output = Dense(1, activation='sigmoid', name=output_name)(net)
    return net, output


def get_layers(inputs, node_output_dim=256):
    q = deque()
    layers = dict()
    layers[GO_ID] = {'net': inputs}
    for node_id in go[GO_ID]['children']:
        if node_id in func_set:
            q.append((node_id, node_output_dim))
    while len(q) > 0:
        node_id, output_dim = q.popleft()
        parents = get_parents(go, node_id)
        parent_models = []
        for parent_id in parents:
            if parent_id in layers:
                parent_models.append(layers[parent_id]['net'])
        net, output = get_function_node(node_id, parent_models, output_dim)
        layers[node_id] = {'net': net, 'output': output}
        for n_id in go[node_id]['children']:
            if n_id in func_set:
                q.append((n_id, output_dim))
    return layers


def model():
    # set parameters:
    batch_size = 64
    nb_epoch = 10
    nb_classes = len(functions)
    start_time = time.time()
    logging.info("Loading Data")
    train, val, test, test_gos = load_data()
    train_data, train_labels = train
    val_data, val_labels = val
    test_data, test_labels = test
    logging.info("Data loaded in %d sec" % (time.time() - start_time))
    logging.info("Building the model")
    inputs = Input(shape=(MAXLEN,), dtype='int32', name='input1')
    inputs2 = Input(shape=(REPLEN,), dtype='float32', name='input2')
    feature_model = get_feature_model()(inputs)
    merged = merge([feature_model, inputs2], mode='concat', name='merged')
    layers = get_layers(merged)
    output_models = []
    for i in range(len(functions)):
        output_models.append(layers[functions[i]]['output'])
    model = Model(input=[inputs, inputs2], output=output_models)
    logging.info('Model built in %d sec' % (time.time() - start_time))
    logging.info('Saving the model')
    model_json = model.to_json()
    with open(DATA_ROOT + 'model_' + FUNCTION + ORG + '.json', 'w') as f:
        f.write(model_json)
    logging.info('Compiling the model')
    optimizer = RMSprop()
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy')

    model_path = DATA_ROOT + 'model_weights_' + FUNCTION + ORG + '.h5'
    last_model_path = DATA_ROOT + 'model_weights_' + FUNCTION + ORG + '.last.h5'

    checkpointer = ModelCheckpoint(
        filepath=model_path,
        verbose=1, save_best_only=True, save_weights_only=True)
    logging.info(
        'Compilation finished in %d sec' % (time.time() - start_time))
    logging.info('Starting training the model')

    train_generator = DataGenerator(batch_size, nb_classes)
    train_generator.fit(train_data, train_labels)
    valid_generator = DataGenerator(batch_size, nb_classes)
    valid_generator.fit(val_data, val_labels)
    test_generator = DataGenerator(batch_size, nb_classes)
    test_generator.fit((test_data[0], test_data[1]), test_labels)

    model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_data[0]),
        nb_epoch=nb_epoch,
        validation_data=valid_generator,
        nb_val_samples=len(val_data[0]),
        max_q_size=batch_size,
        callbacks=[checkpointer])
    model.save_weights(last_model_path)

    logging.info('Loading weights')
    model.load_weights(model_path)
    output_test = []
    for i in range(len(functions)):
        output_test.append(np.array(test_labels[i]))
    preds = model.predict_generator(
        test_generator, val_samples=len(test_data[0]))
    for i in xrange(len(preds)):
        preds[i] = preds[i].reshape(-1, 1)
    preds = np.concatenate(preds, axis=1)
    f, p, r = compute_performance(preds, test_labels, test_gos)
    roc_auc = compute_roc(preds, test_labels)
    logging.info('F measure: \t %f %f %f' % (f, p, r))
    logging.info('ROC AUC: \t %f ' % (roc_auc, ))
    logging.info('Done in %d sec' % (time.time() - start_time))


def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_performance(preds, labels, gos):
    preds = (preds > 0.5).astype(np.int32)
    total = 0
    f = 0.0
    p = 0.0
    r = 0.0
    for i in range(labels.shape[0]):
        tp = np.sum(preds[i, :] * labels[i, :])
        fp = np.sum(preds[i, :]) - tp
        fn = np.sum(labels[i, :]) - tp
        if tp == 0 and fp == 0 and fn == 0:
            continue
        total += 1
        if tp != 0:
            precision = tp / (1.0 * (tp + fp))
            recall = tp / (1.0 * (tp + fn))
            p += precision
            r += recall
            f += 2 * precision * recall / (precision + recall)
    return f / total, p / total, r / total


def print_report(report, go_id):
    with open(DATA_ROOT + 'reports.txt', 'a') as f:
        f.write('Classification report for ' + go_id + '\n')
        f.write(report + '\n')


if __name__ == '__main__':
    main()
