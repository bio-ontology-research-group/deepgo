#!/usr/bin/env python

"""
python nn_hierarchical_network.py
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
from keras.optimizers import Adam, RMSprop, Adadelta
from sklearn.metrics import classification_report
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    get_parents,
    DataGenerator,
    FUNC_DICT,
    MyCheckpoint,
    save_model_weights,
    load_model_weights)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras import backend as K
import sys
from collections import deque
import time
import logging
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from scipy.spatial import distance
from multiprocessing import Pool

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
sys.setrecursionlimit(100000)

DATA_ROOT = 'data/swiss/'
MAXLEN = 1000
REPLEN = 256
ind = 0


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
    default=None,
    help='Organism id for filtering test set')
def main(function, device, org):
    global FUNCTION
    FUNCTION = function
    global GO_ID
    GO_ID = FUNC_DICT[FUNCTION]
    global go
    go = get_gene_ontology('go.obo')
    global ORG
    ORG = org
    func_df = pd.read_pickle(DATA_ROOT + FUNCTION + '.pkl')
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
    global node_names
    node_names = set()
    with tf.device('/' + device):
        model()


def load_data(split=0.7):
    df = pd.read_pickle(DATA_ROOT + 'train' + '-' + FUNCTION + '.pkl')
    n = len(df)
    index = np.arange(n)
    valid_n = int(n * 0.8)
    train_df = df.iloc[index[:valid_n]]
    valid_df = df.iloc[index[valid_n:]]
    test_df = pd.read_pickle(DATA_ROOT + 'test' + '-' + FUNCTION + '.pkl')
    if ORG is not None:
        logging.info('Unfiltered test size: %d' % len(test_df))
        test_df = test_df[test_df['orgs'] == ORG]
        logging.info('Filtered test size: %d' % len(test_df))

    def reshape(values):
        values = np.hstack(values).reshape(
            len(values), len(values[0]))
        return values

    def get_values(data_frame):
        labels = reshape(data_frame['labels'].values)
        ngrams = sequence.pad_sequences(
            data_frame['ngrams'].values, maxlen=MAXLEN)
        ngrams = reshape(ngrams)
        return ngrams, labels

    train = get_values(train_df)
    valid = get_values(valid_df)
    test = get_values(test_df)

    return train, valid, test, test_df


def get_feature_model():
    embedding_dims = 128
    max_features = 8001
    model = Sequential()
    model.add(Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN,
        dropout=0.2))
    model.add(Convolution1D(
        nb_filter=16,
        filter_length=64,
        border_mode='valid',
        activation='relu',
        subsample_length=1))
    model.add(MaxPooling1D(pool_length=32, stride=16))
    model.add(Flatten())
    return model


def merge_outputs(outputs, name):
    if len(outputs) == 1:
        return outputs[0]
    return merge(outputs, mode='concat', name=name, concat_axis=1)


def merge_nets(nets, name):
    if len(nets) == 1:
        return nets[0]
    return merge(nets, mode='sum', name=name)


def get_node_name(go_id, unique=False):
    name = go_id.split(':')[1]
    if not unique:
        return name
    if name not in node_names:
        node_names.add(name)
        return name
    i = 1
    while (name + '_' + str(i)) in node_names:
        i += 1
    name = name + '_' + str(i)
    node_names.add(name)
    return name


def get_function_node(name, inputs):
    output_name = name + '_out'
    net = Dense(256, name=name, activation='relu')(inputs)
    output = Dense(1, name=output_name, activation='sigmoid')(net)
    return net, output


def get_layers_recursive(inputs, node_output_dim=256):
    layers = dict()
    name = get_node_name(GO_ID)

    def dfs(node_id, inputs):
        name = get_node_name(node_id, unique=True)
        net, output = get_function_node(name, inputs, node_output_dim)
        childs = [
            n_id for n_id in go[node_id]['children'] if n_id in func_set]
        if node_id not in layers:
            layers[node_id] = {'outputs': [output]}
        else:
            layers[node_id]['outputs'].append(output)
        for ch_id in childs:
            dfs(ch_id, net)

    for node_id in go[GO_ID]['children']:
        if node_id in func_set:
            dfs(node_id, inputs)

    for node_id in functions:
        childs = get_go_set(go, node_id).intersection(func_set)
        if len(childs) == 0:
            if len(layers[node_id]['outputs']) == 1:
                layers[node_id]['output'] = layers[node_id]['outputs'][0]
            else:
                name = get_node_name(node_id, unique=True)
                output = merge(
                    layers[node_id]['outputs'], mode='max', name=name)
                layers[node_id]['output'] = output
        else:
            outputs = layers[node_id]['outputs']
            for ch_id in childs:
                outputs += layers[ch_id]['outputs']
            name = get_node_name(node_id, unique=True)
            output = merge(
                outputs, mode='max', name=name)
            layers[node_id]['output'] = output
    return layers


def get_layers(inputs):
    q = deque()
    layers = {}
    name = get_node_name(GO_ID)
    layers[GO_ID] = {'net': inputs}
    for node_id in go[GO_ID]['children']:
        if node_id in func_set:
            q.append((node_id, inputs))
    while len(q) > 0:
        node_id, net = q.popleft()
        parent_nets = [inputs]
        for p_id in get_parents(go, node_id):
            if p_id in func_set:
                parent_nets.append(layers[p_id]['net'])
        if len(parent_nets) > 1:
            name = get_node_name(node_id) + '_parents'
            net = merge(
                parent_nets, mode='concat', concat_axis=1, name=name)
        name = get_node_name(node_id)
        net, output = get_function_node(name, net)
        if node_id not in layers:
            layers[node_id] = {'net': net, 'output': output}
            for n_id in go[node_id]['children']:
                if n_id in func_set and n_id not in layers:
                    ok = True
                    for p_id in get_parents(go, n_id):
                        if p_id in func_set and p_id not in layers:
                            ok = False
                    if ok:
                        q.append((n_id, net))

    for node_id in functions:
        childs = set(go[node_id]['children']).intersection(func_set)
        if len(childs) > 0:
            outputs = [layers[node_id]['output']]
            for ch_id in childs:
                outputs.append(layers[ch_id]['output'])
            name = get_node_name(node_id) + '_max'
            layers[node_id]['output'] = merge(
                outputs, mode='max', name=name)
    return layers


def model():
    # set parameters:
    batch_size = 128
    nb_epoch = 100
    nb_classes = len(functions)
    start_time = time.time()
    logging.info("Loading Data")
    train, val, test, test_df = load_data()
    test_gos = test_df['gos'].values
    train_data, train_labels = train
    val_data, val_labels = val
    test_data, test_labels = test
    logging.info("Data loaded in %d sec" % (time.time() - start_time))
    logging.info("Training data size: %d" % len(train_data))
    logging.info("Validation data size: %d" % len(val_data))
    logging.info("Test data size: %d" % len(test_data))
    logging.info("Building the model")
    inputs = Input(shape=(MAXLEN,), dtype='int32', name='input1')
    feature_model = get_feature_model()(inputs)
    layers = get_layers(feature_model)
    output_models = []
    for i in range(len(functions)):
        output_models.append(layers[functions[i]]['output'])
    net = merge(output_models, mode='concat', concat_axis=1)
    # net = Dense(nb_classes * 2, activation='relu')(feature_model)
    # net = Dense(nb_classes, activation='sigmoid')(net)
    # net = Activation('sigmoid')(net)
    model = Model(input=inputs, output=net)
    logging.info('Model built in %d sec' % (time.time() - start_time))
    logging.info('Saving the model')
    model_json = model.to_json()
    with open(DATA_ROOT + 'model_seq_' + FUNCTION + '.json', 'w') as f:
        f.write(model_json)
    logging.info('Compiling the model')
    optimizer = RMSprop()

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy')

    pre_model_path = DATA_ROOT + 'pre_model_seq_weights_' + FUNCTION + '.pkl'
    model_path = DATA_ROOT + 'model_seq_weights_' + FUNCTION + '.h5'
    checkpointer = ModelCheckpoint(
        filepath=model_path,
        verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    logging.info(
        'Compilation finished in %d sec' % (time.time() - start_time))

    # logging.info('Loading pretrained weights')
    # load_model_weights(model, pre_model_path)

    logging.info('Starting training the model')

    train_generator = DataGenerator(batch_size, nb_classes)
    train_generator.fit(train_data, train_labels)
    valid_generator = DataGenerator(batch_size, nb_classes)
    valid_generator.fit(val_data, val_labels)
    test_generator = DataGenerator(batch_size, nb_classes)
    test_generator.fit(test_data, test_labels)
    # model.fit_generator(
    #     train_generator,
    #     samples_per_epoch=len(train_data),
    #     nb_epoch=nb_epoch,
    #     validation_data=valid_generator,
    #     nb_val_samples=len(val_data),
    #     max_q_size=batch_size,
    #     callbacks=[checkpointer, earlystopper])

    logging.info('Loading weights')
    load_model_weights(model, model_path)

    # model.save(DATA_ROOT + 'model_%s.h5' % FUNCTION)

    preds = model.predict_generator(
        test_generator, val_samples=len(test_data))

    logging.info(preds.shape)
    incon = 0
    # for i in xrange(len(test_data)):
    #     for j in xrange(len(functions)):
    #         childs = set(go[functions[j]]['children']).intersection(func_set)
    #         ok = True
    #         for n_id in childs:
    #             if preds[i, j] < preds[i, go_indexes[n_id]]:
    #                 preds[i, j] = preds[i, go_indexes[n_id]]
    #                 ok = False
    #         if not ok:
    #             incon += 1
    f, p, r, preds_max = compute_performance(preds, test_labels, test_gos)
    roc_auc = compute_roc(preds, test_labels)
    logging.info('Fmax measure: \t %f %f %f' % (f, p, r))
    logging.info('ROC AUC: \t %f ' % (roc_auc, ))
    logging.info('Inconsistent predictions: %d' % incon)
    logging.info('Saving the predictions')
    proteins = test_df['proteins']
    predictions = list()
    for i in xrange(preds_max.shape[0]):
        predictions.append(preds_max[i])
    df = pd.DataFrame(
        {
            'proteins': proteins, 'predictions': predictions,
            'gos': test_df['gos'], 'labels': test_df['labels']})
    df.to_pickle(DATA_ROOT + 'test-' + FUNCTION + '-preds-seq.pkl')
    logging.info('Done in %d sec' % (time.time() - start_time))

    function_centric_performance(functions, preds.T, test_labels.T)


def function_centric_performance(functions, preds, labels):
    preds = np.round(preds, 2)
    for i in xrange(len(functions)):
        f_max = 0
        p_max = 0
        r_max = 0
        x = list()
        y = list()
        for t in xrange(1, 100):
            threshold = t / 100.0
            predictions = (preds[i, :] > threshold).astype(np.int32)
            tp = np.sum(predictions * labels[i, :])
            fp = np.sum(predictions) - tp
            fn = np.sum(labels[i, :]) - tp
            sn = tp / (1.0 * np.sum(labels[i, :]))
            sp = np.sum((predictions ^ 1) * (labels[i, :] ^ 1))
            sp /= 1.0 * np.sum(labels[i, :] ^ 1)
            fpr = 1 - sp
            x.append(fpr)
            y.append(sn)
            precision = tp / (1.0 * (tp + fp))
            recall = tp / (1.0 * (tp + fn))
            f = 2 * precision * recall / (precision + recall)
            if f_max < f:
                f_max = f
                p_max = precision
                r_max = recall
        num_prots = np.sum(labels[i, :])
        roc_auc = auc(x, y)
        print('%s %f %f %f %d %f' % (
            functions[i], f_max, p_max, r_max, num_prots, roc_auc))


def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_performance(preds, labels, gos):
    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    for t in xrange(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            # all_gos = set()
            # for go_id in gos[i]:
            #     if go_id in all_functions:
            #         all_gos |= get_anchestors(go, go_id)
            # all_gos.discard(GO_ID)
            # all_gos -= func_set
            # fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if p_total == 0:
            continue
        r /= total
        p /= p_total
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f_max < f:
                f_max = f
                p_max = p
                r_max = r
                predictions_max = predictions
    return f_max, p_max, r_max, predictions_max


if __name__ == '__main__':
    main()
