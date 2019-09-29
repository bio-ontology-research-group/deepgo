#!/usr/bin/env python

"""
python nn_hierarchical_network.py
"""

import numpy as np
import pandas as pd
import click as ck
from keras.models import Sequential, Model, load_model
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
    load_model_weights,
    get_ipro)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras import backend as K
import sys
from collections import deque
import time
import logging
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from multiprocessing import Pool

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
sys.setrecursionlimit(100000)

DATA_ROOT = 'data/all/'
MAXLEN = 1000
REPLEN = 256
ind = 0


@ck.command()
@ck.option(
    '--device',
    default='gpu:0',
    help='GPU or CPU device id')
@ck.option(
    '--org',
    default=None,
    help='Organism id for filtering test set')
@ck.option('--train', is_flag=True)
def main(device, org, train):
    global GO_IDS
    GO_IDS = list(FUNC_DICT.values())
    global go
    go = get_gene_ontology('go.obo')
    global ORG
    ORG = org
    func_df = pd.read_pickle(DATA_ROOT + 'bp.pkl')
    global functions
    functions = func_df['functions'].values
    func_df = pd.read_pickle(DATA_ROOT + 'mf.pkl')
    functions = np.concatenate((functions, func_df['functions'].values))
    func_df = pd.read_pickle(DATA_ROOT + 'cc.pkl')
    functions = np.concatenate((functions, func_df['functions'].values))
    global func_set
    func_set = set(functions)
    global all_functions
    all_functions = (
        get_go_set(go, GO_IDS[0])
        | get_go_set(go, GO_IDS[1])
        | get_go_set(go, GO_IDS[2]))
    logging.info('Functions: %d' % (len(functions), ))
    if ORG is not None:
        logging.info('Organism %s' % ORG)
    global go_indexes
    go_indexes = dict()
    for ind, go_id in enumerate(functions):
        go_indexes[go_id] = ind
    global node_names
    node_names = set()
    with tf.device('/' + device):
        params = {
            'fc_output': 1024,
            'learning_rate': 0.001,
            'embedding_dims': 128,
            'embedding_dropout': 0.2,
            'filter_length': 128,
            'nb_filter': 32,
            'pool_length': 64,
            'stride': 32
        }
        model(params, is_train=train)
    #     for dims in [64, 128, 256, 512]:
    #         for nb_filter in [16, 32, 64, 128]:
    #             params['embedding_dims'] = dims
    #             params['nb_filter'] = nb_filter
    #             f = model(params, is_train=train)
    #             print(dims, nb_filter, f)
    # performanc_by_interpro()


def load_data(org=None):

    df = pd.read_pickle(DATA_ROOT + 'train.pkl')

    test_df = pd.read_pickle(DATA_ROOT + 'test.pkl')
    # df = pd.concat([df, test_df], ignore_index=True)
    n = len(df)
    print(n)
    index = df.index.values
    valid_n = int(n * 0.8)
    train_df = df.loc[index[:valid_n]]
    valid_df = df.loc[index[valid_n:]]

    # test_df = valid_df
    if org is not None:
        logging.info('Unfiltered test size: %d' % len(test_df))
        test_df = test_df[test_df['orgs'] == org]
        logging.info('Filtered test size: %d' % len(test_df))

    # Filter by type
    # org_df = pd.read_pickle('data/eukaryotes.pkl')
    # orgs = org_df['orgs']
    # test_df = test_df[test_df['orgs'].isin(orgs)]

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
        ngrams = sequence.pad_sequences(
            data_frame['ngrams'].values, maxlen=MAXLEN)
        ngrams = reshape(ngrams)
        rep = reshape(data_frame['embeddings'].values)
        data = (ngrams, rep)
        return data, labels

    train = get_values(train_df)
    valid = get_values(valid_df)
    test = get_values(test_df)

    return train, valid, test, train_df, valid_df, test_df


def get_feature_model(params):
    embedding_dims = params['embedding_dims']
    max_features = 8001
    model = Sequential()
    model.add(Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN,
        dropout=params['embedding_dropout']))
    model.add(Convolution1D(
        nb_filter=params['nb_filter'],
        filter_length=params['filter_length'],
        border_mode='valid',
        activation='relu',
        subsample_length=1))
    model.add(MaxPooling1D(
        pool_length=params['pool_length'], stride=params['stride']))
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
    # net = Dense(256, name=name, activation='relu')(inputs)
    output = Dense(1, name=output_name, activation='sigmoid')(inputs)
    return output, output


def get_layers(inputs):
    q = deque()
    layers = {}
    layers[GO_IDS[0]] = {'net': inputs}
    layers[GO_IDS[1]] = {'net': inputs}
    layers[GO_IDS[2]] = {'net': inputs}
    for node_id in go[GO_IDS[0]]['children']:
        if node_id in func_set:
            q.append((node_id, inputs))
    for node_id in go[GO_IDS[1]]['children']:
        if node_id in func_set:
            q.append((node_id, inputs))
    for node_id in go[GO_IDS[2]]['children']:
        if node_id in func_set:
            q.append((node_id, inputs))
    
    while len(q) > 0:
        node_id, net = q.popleft()
        parent_nets = [inputs]
        # for p_id in get_parents(go, node_id):
        #     if p_id in func_set:
        #         parent_nets.append(layers[p_id]['net'])
        # if len(parent_nets) > 1:
        #     name = get_node_name(node_id) + '_parents'
        #     net = merge(
        #         parent_nets, mode='concat', concat_axis=1, name=name)
        name = get_node_name(node_id)
        net, output = get_function_node(name, inputs)
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


def get_model(params):
    logging.info("Building the model")
    inputs = Input(shape=(MAXLEN,), dtype='int32', name='input1')
    inputs2 = Input(shape=(REPLEN,), dtype='float32', name='input2')
    feature_model = get_feature_model(params)(inputs)
    merged = merge(
        [feature_model, inputs2], mode='concat',
        concat_axis=1, name='merged')
    net = Dense(params['fc_output'], activation='relu')(merged)
    layers = get_layers(net)
    output_models = []
    for i in range(len(functions)):
        output_models.append(layers[functions[i]]['output'])
    net = merge(output_models, mode='concat', concat_axis=1)
    # net = Dense(1024, activation='relu')(merged)
    # net = Dense(len(functions), activation='sigmoid')(net)
    model = Model(input=[inputs, inputs2], output=net)
    logging.info('Compiling the model')
    optimizer = RMSprop(lr=params['learning_rate'])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy')
    logging.info(
        'Compilation finished')
    return model


def model(params, batch_size=128, nb_epoch=6, is_train=True):
    # set parameters:
    nb_classes = len(functions)
    start_time = time.time()
    logging.info("Loading Data")
    train, val, test, train_df, valid_df, test_df = load_data()
    train_df = pd.concat([train_df, valid_df])
    test_gos = test_df['gos'].values
    train_data, train_labels = train
    val_data, val_labels = val
    test_data, test_labels = test
    logging.info("Data loaded in %d sec" % (time.time() - start_time))
    logging.info("Training data size: %d" % len(train_data[0]))
    logging.info("Validation data size: %d" % len(val_data[0]))
    logging.info("Test data size: %d" % len(test_data[0]))

    model_path = (DATA_ROOT + 'model.h5')
                  # '-' + str(params['embedding_dims']) +
                  # '-' + str(params['nb_filter']) + '.h5')
    checkpointer = ModelCheckpoint(
        filepath=model_path,
        verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    logging.info('Starting training the model')

    train_generator = DataGenerator(batch_size, nb_classes)
    train_generator.fit(train_data, train_labels)
    valid_generator = DataGenerator(batch_size, nb_classes)
    valid_generator.fit(val_data, val_labels)
    test_generator = DataGenerator(batch_size, nb_classes)
    test_generator.fit(test_data, test_labels)

    if is_train:
        model = get_model(params)
        model.fit_generator(
            train_generator,
            samples_per_epoch=len(train_data[0]),
            nb_epoch=nb_epoch,
            validation_data=valid_generator,
            nb_val_samples=len(val_data[0]),
            max_q_size=batch_size,
            callbacks=[checkpointer, earlystopper])

    logging.info('Loading best model')
    model = load_model(model_path)

    # orgs = ['9606', '10090', '10116', '7227', '7955',
    #         '559292', '3702', '284812', '6239',
    #         '83333', '83332', '224308', '208964']
    # for org in orgs:
    #     logging.info('Predicting for %s' % (org,))
    #     train, val, test, train_df, valid_df, test_df = load_data(org=org)
    #     test_data, test_labels = test
    #     test_gos = test_df['gos'].values
    #     test_generator = DataGenerator(batch_size, nb_classes)
    #     test_generator.fit(test_data, test_labels)
    preds = model.predict_generator(
        test_generator, val_samples=len(test_data[0]))
    logging.info('Computing performance')
    f, p, r, t, preds_max = compute_performance(preds, test_labels, test_gos)
    roc_auc = compute_roc(preds, test_labels)
    mcc = compute_mcc(preds_max, test_labels)
    logging.info('Fmax measure: \t %f %f %f %f' % (f, p, r, t))
    logging.info('ROC AUC: \t %f ' % (roc_auc, ))
    logging.info('MCC: \t %f ' % (mcc, ))
    print(('%.3f & %.3f & %.3f & %.3f & %.3f' % (
        f, p, r, roc_auc, mcc)))
    # return f
    # logging.info('Inconsistent predictions: %d' % incon)
    # logging.info('Saving the predictions')
    proteins = test_df['proteins']
    predictions = list()
    for i in range(preds_max.shape[0]):
        predictions.append(preds_max[i])
    df = pd.DataFrame(
        {
            'proteins': proteins, 'predictions': predictions,
            'gos': test_df['gos'], 'labels': test_df['labels']})
    df.to_pickle(DATA_ROOT + 'test-' + FUNCTION + '-preds.pkl')
    # logging.info('Done in %d sec' % (time.time() - start_time))

    # function_centric_performance(functions, preds.T, test_labels.T)


def load_prot_ipro():
    proteins = list()
    ipros = list()
    with open(DATA_ROOT + 'swissprot_ipro.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            if len(it) != 3:
                continue
            prot = it[1]
            iprs = set(it[2].split(';'))
            proteins.append(prot)
            ipros.append(iprs)
    return pd.DataFrame({'proteins': proteins, 'ipros': ipros})


def performanc_by_interpro():
    pred_df = pd.read_pickle(DATA_ROOT + 'test-' + FUNCTION + '-preds.pkl')
    ipro_df = load_prot_ipro()
    df = pred_df.merge(ipro_df, on='proteins', how='left')
    ipro = get_ipro()

    def reshape(values):
        values = np.hstack(values).reshape(
            len(values), len(values[0]))
        return values

    for ipro_id in ipro:
        if len(ipro[ipro_id]['parents']) > 0:
            continue
        labels = list()
        predictions = list()
        gos = list()
        for i, row in df.iterrows():
            if not isinstance(row['ipros'], set):
                continue
            if ipro_id in row['ipros']:
                labels.append(row['labels'])
                predictions.append(row['predictions'])
                gos.append(row['gos'])
        pr = 0
        rc = 0
        total = 0
        p_total = 0
        for i in range(len(labels)):
            tp = np.sum(labels[i] * predictions[i])
            fp = np.sum(predictions[i]) - tp
            fn = np.sum(labels[i]) - tp
            all_gos = set()
            for go_id in gos[i]:
                if go_id in all_functions:
                    all_gos |= get_anchestors(go, go_id)
            all_gos.discard(GO_ID)
            all_gos -= func_set
            fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                pr += precision
                rc += recall
        if total > 0 and p_total > 0:
            rc /= total
            pr /= p_total
            if pr + rc > 0:
                f = 2 * pr * rc / (pr + rc)
                print(('%s\t%d\t%f\t%f\t%f' % (
                    ipro_id, len(labels), f, pr, rc)))


def function_centric_performance(functions, preds, labels):
    preds = np.round(preds, 2)
    for i in range(len(functions)):
        f_max = 0
        p_max = 0
        r_max = 0
        x = list()
        y = list()
        for t in range(1, 100):
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
        print(('%s %f %f %f %d %f' % (
            functions[i], f_max, p_max, r_max, num_prots, roc_auc)))


def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(preds, labels):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def compute_performance(preds, labels, gos):
    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
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
            all_gos = set()
            for go_id in gos[i]:
                if go_id in all_functions:
                    all_gos |= get_anchestors(go, go_id)
            for g_id in GO_IDS:
                all_gos.discard(g_id)
            all_gos -= func_set
            fn += len(all_gos)
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
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max


def get_gos(pred):
    mdist = 1.0
    mgos = None
    for i in range(len(labels_gos)):
        labels, gos = labels_gos[i]
        dist = distance.cosine(pred, labels)
        if mdist > dist:
            mdist = dist
            mgos = gos
    return mgos


def compute_similarity_performance(train_df, test_df, preds):
    logging.info("Computing similarity performance")
    logging.info("Training data size %d" % len(train_df))
    train_labels = train_df['labels'].values
    train_gos = train_df['gos'].values
    global labels_gos
    labels_gos = list(zip(train_labels, train_gos))
    p = Pool(64)
    pred_gos = p.map(get_gos, preds)
    total = 0
    p = 0.0
    r = 0.0
    f = 0.0
    test_gos = test_df['gos'].values
    for gos, tgos in zip(pred_gos, test_gos):
        preds = set()
        test = set()
        for go_id in gos:
            if go_id in all_functions:
                preds |= get_anchestors(go, go_id)
        for go_id in tgos:
            if go_id in all_functions:
                test |= get_anchestors(go, go_id)
        tp = len(preds.intersection(test))
        fp = len(preds - test)
        fn = len(test - preds)
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
