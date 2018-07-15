#!/usr/bin/env python

"""
python nn_hierarchical_network.py
"""

import numpy as np
import pandas as pd
import click as ck
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import (
    Dense, Dropout, Activation, Input,
    Flatten, Highway, BatchNormalization)
from keras.layers.merge import concatenate, maximum
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import (
    Conv1D, MaxPooling1D)
from keras.optimizers import Adam, RMSprop, Adadelta
from sklearn.metrics import classification_report
from utils import (
    get_gene_ontology,
    get_go_set,
    get_anchestors,
    get_parents,
    FUNC_DICT,
    get_ipro)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import sys
from collections import deque
import time
import logging
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
import math
import string
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
sys.setrecursionlimit(100000)

DATA_ROOT = 'data/binary/'
MAXLEN = 2000

class DFGenerator(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def fit(self, df):
        self.start = 0
        self.size = len(df)
        self.df = df
    
    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            df = self.df.iloc[batch_index]
            data_seq = np.zeros((len(df), MAXLEN), dtype=np.int32)
            data_net = np.zeros((len(df), 256), dtype=np.float32)
            ipros = np.zeros((len(df), len(interpros)), dtype=np.float32)
            labels = np.zeros((len(df), nb_classes), dtype=np.int32)
            for i, row in enumerate(df.itertuples()):
                st = 0
                if hasattr(row, 'starts'):
                    st = row.starts
                data_seq[i, st:(st + len(row.ngrams))] = row.ngrams
                if isinstance(row.embeddings, np.ndarray):
                    data_net[i, :] = row.embeddings
                if isinstance(row.interpros, list):
                    for ipro_id in row.interpros:
                        if ipro_id in ipro_indexes:
                            ipros[i, ipro_indexes[ipro_id]] = 1
            
                for go_id in row.functions:
                    if go_id in go_indexes:
                        labels[i, go_indexes[go_id]] = 1
            self.start += self.batch_size
            data = [data_seq, ipros, data_net]
            return (data, labels)
        else:
            self.reset()
            return self.next()


@ck.command()
@ck.option(
    '--device',
    default='gpu:0',
    help='GPU or CPU device id')
@ck.option(
    '--org',
    default=None,
    help='Organism id for filtering test set')
@ck.option(
    '--model-file',
    default=(DATA_ROOT + 'model'),
    help='Model filename')
@ck.option('--is-train', is_flag=True)
@ck.option(
    '--batch-size',
    default=128,
    help='Batch size for training and testing')
@ck.option(
    '--epochs',
    default=12,
    help='Number of epochs to train')
def main(device, org, model_file, is_train, batch_size, epochs):
    global go
    go = get_gene_ontology(DATA_ROOT + 'go.obo', with_rels=True)
    func_df = pd.read_pickle(DATA_ROOT + 'functions.pkl')
    global functions
    functions = func_df['functions'].values
    global func_set
    func_set = set(functions)
    global nb_classes
    nb_classes = len(functions)
    logging.info('Functions: %d' % nb_classes)
    if org is not None:
        logging.info('Organism %s' % ORG)
    global go_indexes
    go_indexes = {}
    for ind, go_id in enumerate(functions):
        go_indexes[go_id] = ind
    global names
    names = {}
    global digits
    digits = string.digits + string.ascii_letters

    global ipro_indexes
    ipro_indexes = {}
    global interpros
    interpros = list()
    with open(DATA_ROOT + 'interpros.list') as f:
        for i in range(10000):
            it = next(f).split('\t')
            ipro_indexes[it[0]] = i
            interpros.append(it[0])
    
    train_df, valid_df, test_df = load_data(org=org)
    with tf.device('/' + device):
        if is_train:
            train_model(train_df, valid_df, model_file, batch_size, epochs)
        test_model(test_df, model_file, batch_size)

def load_data(org):
    train_df = pd.read_pickle(DATA_ROOT + 'train.pkl')
    test_df = pd.read_pickle(DATA_ROOT + 'test.pkl')
    n = len(train_df)
    index = np.arange(n)
    train_n = int(n * 0.8)
    valid_df = train_df.iloc[index[train_n:]]
    train_df = train_df.iloc[index[:train_n]]
    if org is not None:
        logging.info('Unfiltered test size: %d' % len(test_df))
        test_df = test_df[test_df['orgs'] == org]
        logging.info('Filtered test size: %d' % len(test_df))

    # Filter by type
    # org_df = pd.read_pickle('data/prokaryotes.pkl')
    # orgs = org_df['orgs']
    # test_df = test_df[test_df['orgs'].isin(orgs)]
    def augment(df):
        functions = list()
        ngrams = list()
        embeddings = list()
        starts = list()
        for i, row in enumerate(df.itertuples()):
            st = np.random.randint((MAXLEN - len(row.ngrams)), size=10)
            for s in st:
                functions.append(row.functions)
                ngrams.append(row.ngrams)
                embeddings.append(row.embeddings)
                starts.append(s)
        df = pd.DataFrame({
            'functions': functions, 'ngrams': ngrams,
            'embeddings': embeddings, 'starts': starts})
        index = np.arange(len(df))
        np.random.seed(seed=10)
        np.random.shuffle(index)
        return df.iloc[index]

    return train_df, valid_df, test_df


def get_feature_model():
    embedding_dims = 32
    max_features = 8001
    model = Sequential()
    model.add(Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN))
    model.add(Conv1D(
        filters=1,
        kernel_size=3,
        padding='valid',
        strides=1))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.3))
    # model.add(Conv1D(
    #     filters=64,
    #     kernel_size=7,
    #     padding='valid',
    #     dilation_rate=2,
    #     strides=1))
    # model.add(MaxPooling1D(pool_size=3))
    # # model.add(Dropout(0.3))
    # model.add(Conv1D(
    #     filters=64,
    #     kernel_size=7,
    #     padding='valid',
    #     dilation_rate=2,
    #     strides=1))
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Conv1D(
    #     filters=64,
    #     kernel_size=7,
    #     padding='valid',
    #     dilation_rate=2,
    #     strides=1))
    # model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    return model


def get_node_name(go_id):
    name = go_id.split(':')[1]
    return name
    # if go_id in names:
    #     return names[go_id]
    # n = len(names)
    # b = len(digits)
    # name = ''
    # while n > 0:
    #     name += digits[n % b]
    #     n = n // b
    # names[go_id] = name
    # return name


def get_function_node(name, inputs, embed, ipros):
    model_file = 'data/binary/model_GO_' + name + '.h5'
    # if os.path.exists(model_file):
    #     model = load_model(model_file)
    #     model.name = 'model_' + name
    #     print(name, model.input)
    #     net = model([inputs, embed])
    #     return net
    embedding_dims = 48
    max_features = 8001
    net = Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN,
        name=(name + '_e'))(inputs)
    net = Conv1D(
        filters=128,
        kernel_size=15,
        padding='valid',
        strides=1,
        name=(name + '_c0'))(net)
    net = Conv1D(
        filters=16,
        kernel_size=15,
        padding='valid',
        strides=1,
        name=(name + '_c1'))(net)

    net = Flatten()(net)
    net = concatenate([net, ipros, embed], axis=1)
    # net = ipros
    net = Dense(1, name=name, activation='sigmoid')(net)
    return net


def get_layers(inputs):
    q = deque()
    layers = {}
    
    for fn in FUNC_DICT:
        layers[FUNC_DICT[fn]] = {'net': inputs}
        for node_id in go[FUNC_DICT[fn]]['children']:
            if node_id in func_set:
                q.append((node_id, inputs))
    while len(q) > 0:
        node_id, net = q.popleft()
        parent_nets = [inputs]
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
            name = get_node_name(node_id + '_')
            layers[node_id]['output'] = maximum(outputs, name=name)
    return layers


def get_model():
    logging.info("Building the model")
    input_seq = Input(shape=(MAXLEN,), dtype='int32', name='seq')
    input_embed = Input(shape=(256,), dtype='float32', name='net')
    input_ipros = Input(shape=(len(interpros),), dtype='float32', name='ipros')
    # net = get_feature_model()(input_seq)
    # net = concatenate([net, input_net], axis=1)
    # layers = get_layers(net)
    output_models = []
    for i in range(len(functions)):
        name = get_node_name(functions[i])
        # print(i, functions[i])
        # model_file = 'data/binary/model_GO_' + name + '.h5'
        # if not os.path.exists(model_file):
        #     print('No model found ', i, functions[i])
        output_models.append(
            get_function_node(
                name, input_seq, input_embed, input_ipros))
    net = concatenate(output_models, axis=1)
    # net = Dense(1024, activation='relu')(merged)
    # net = Dense(nb_classes, activation='sigmoid')(net)
    # encoder = load_model('model_encoder.h5')
    # inputs = encoder.inputs
    # features = encoder.layers[1](inputs)
    # features = Flatten()(features)
    # encoder.summary()
    # net = Dense(nb_classes, activation='sigmoid')(features)
    # model = Model(encoder.layers[0].output, net)
    
    model = Model([input_seq, input_ipros, input_embed], net)
    # model.load_weights('data/latest/model-pre.h5')
    logging.info('Compiling the model')
    optimizer = RMSprop()
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy')
    logging.info('Compilation finished')
    return model


def train_model(train_df, valid_df, model_file, batch_size, epochs):
    nb_classes = len(functions)
    start_time = time.time()
    logging.info("Training data size: %d" % len(train_df))
    logging.info("Validation data size: %d" % len(valid_df))

    checkpointer = ModelCheckpoint(
        filepath=model_file + '.h5',
        verbose=1, save_best_only=True,
        save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=16, verbose=1)

    logging.info('Starting training the model')

    train_generator = DFGenerator(batch_size)
    train_generator.fit(train_df)
    valid_generator = DFGenerator(batch_size)
    valid_generator.fit(valid_df)

    valid_steps = int(math.ceil(len(valid_df) / batch_size))
    train_steps = int(math.ceil(len(train_df) / batch_size))

    model = get_model()
    model_json = model.to_json()
    f = open(model_file + '.json', 'w')
    f.write(model_json)
    f.close()
    model.save_weights(model_file + '_init.h5')
    model.summary()
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        max_queue_size=batch_size,
        workers=12,
        callbacks=[checkpointer, earlystopper])

    
def test_model(test_df, model_file, batch_size):
    generator = DFGenerator(batch_size)
    generator.fit(test_df)
    steps = int(math.ceil(len(test_df) / batch_size))

    logging.info('Loading the model')
    model_json = open(model_file + '.json').read()
    model = model_from_json(model_json)
    model.load_weights(model_file + '.h5')
    logging.info('Compiling model')
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    logging.info('Evaluating model')
    loss = model.evaluate_generator(generator, steps=steps)
    logging.info('Test loss %f' % loss)
    
    logging.info('Predicting')
    generator.reset()
    preds = model.predict_generator(generator, steps=steps)
    nb_classes = len(functions)
    test_labels = np.zeros((len(test_df), nb_classes), dtype=np.int32)
    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.functions:
            if go_id in go_indexes:
                test_labels[i, go_indexes[go_id]] = 1
    mf = get_go_set(go, FUNC_DICT['mf'])
    bp = get_go_set(go, FUNC_DICT['bp'])
    cc = get_go_set(go, FUNC_DICT['cc'])
    bp_index = list()
    mf_index = list()
    cc_index = list()
    for i, go_id in enumerate(functions):
        if go_id in mf:
            mf_index.append(i)
        elif go_id in bp:
            bp_index.append(i)
        elif go_id in cc:
            cc_index.append(i)
    mf_labels = test_labels[:, mf_index]
    bp_labels = test_labels[:, bp_index]
    cc_labels = test_labels[:, cc_index]
    mf_preds = preds[:, mf_index]
    bp_preds = preds[:, bp_index]
    cc_preds = preds[:, cc_index]
    test_gos = test_df['functions'].values
    logging.info('Computing performance')
    f, p, r, t, preds_max = compute_performance(cc_preds, cc_labels, test_gos)
    roc_auc = compute_roc(cc_preds, cc_labels)
    mcc = compute_mcc(preds_max, cc_labels)
    logging.info('Fmax measure: \t %f %f %f %f' % (f, p, r, t))
    logging.info('ROC AUC: \t %f ' % (roc_auc, ))
    logging.info('MCC: \t %f ' % (mcc, ))
    print('%.3f & %.3f & %.3f & %.3f & %.3f' % (
        f, p, r, roc_auc, mcc))
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
            iprs = it[2].split(';')
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
            if not isinstance(row['ipros'], list):
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
                logging.info('%s\t%d\t%f\t%f\t%f' % (
                    ipro_id, len(labels), f, pr, rc))


def function_centric_performance(functions, preds, labels):
    preds = np.round(preds, 2)
    for i in xrange(len(functions)):
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
        print('%s %f %f %f %d %f' % (
            functions[i], f_max, p_max, r_max, num_prots, roc_auc))


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
    print(preds)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    predictions_max = (preds > 0).astype(np.int32)
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
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max


def get_gos(pred):
    mdist = 1.0
    mgos = None
    for i in xrange(len(labels_gos)):
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
    labels_gos = zip(train_labels, train_gos)
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


if __name__ == '__main__':
    main()
