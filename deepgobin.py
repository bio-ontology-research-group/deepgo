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
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
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
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from aaindex import AAINDEX

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
            # data = np.zeros((len(df), MAXLEN, 21), dtype=np.int32)
            data = np.zeros((len(df), MAXLEN), dtype=np.int32)
            ipros = np.zeros((len(df), len(interpros)), dtype=np.float32)
            embed = np.zeros((len(df), 256), dtype=np.float32)
            labels = np.zeros((len(df), 1), dtype=np.int32)
            for i, row in enumerate(df.itertuples()):
                st = 0
                if hasattr(row, 'starts'):
                    st = row.starts
                # indexes = np.array(list(map(lambda x: AAINDEX[x], row.sequences))).astype(np.int32)
                # data[i, np.arange(len(row.sequences)), indexes] = 1
                data[i, st:(st + len(row.ngrams))] = row.ngrams
                if isinstance(row.embeddings, np.ndarray):
                    embed[i] = row.embeddings
                if function in row.functions:
                    labels[i, 0] = 1
                if isinstance(row.interpros, list):
                    for ipro_id in row.interpros:
                        if ipro_id in go_indexes:
                            ipros[i, ipro_indexes[ipro_id]] = 1
            self.start += self.batch_size
            data = [data, embed]
            # data = embed
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
@ck.option(
    '--go-id',
    default=0,
    help='Index of GO term')
def main(device, org, model_file, is_train, batch_size, epochs, go_id):
    global go
    go = get_gene_ontology(DATA_ROOT + 'go.obo', with_rels=True)
    func_df = pd.read_pickle(DATA_ROOT + 'functions.pkl')
    global functions
    functions = func_df['functions'].values
    global function
    function = functions[go_id]
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
    model_file += '_' + function.replace(':', '_')
    
    train_df, valid_df, test_df = load_data(org=org)
    with tf.device('/' + device):
        if is_train:
            train_model(train_df, valid_df, model_file, batch_size, epochs)
        test_model(test_df, model_file, batch_size)

def load_data(org):
    train_df = pd.read_pickle(DATA_ROOT + function.replace(':', '_') + '_train.pkl')
    test_df = pd.read_pickle(DATA_ROOT + function.replace(':', '_') + '_test.pkl')
    n = len(train_df)
    index = np.arange(n)
    np.random.seed(seed=0)
    train_n = int(n * 0.8)
    train_df = df.iloc[index[:train_n]]
    valid_df = df.iloc[index[train_n:]]
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
        interpros = list()
        starts = list()
        for i, row in enumerate(df.itertuples()):
            st = np.random.randint((MAXLEN - len(row.ngrams)), size=20)
            for s in st:
                functions.append(row.functions)
                ngrams.append(row.ngrams)
                embeddings.append(row.embeddings)
                interpros.append(row.interpros)
                starts.append(s)
        df = pd.DataFrame({
            'functions': functions, 'ngrams': ngrams, 'interpros': interpros,
            'embeddings': embeddings, 'starts': starts})
        index = np.arange(len(df))
        np.random.seed(seed=10)
        np.random.shuffle(index)
        return df.iloc[index]

    return train_df, valid_df, test_df


def get_feature_net(seq):
    embedding_dims = 4 
    max_features = 8001
    embed = Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN)(seq)
    net = embed
    return net


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


def get_function_node(name, inputs, embed):
    # net = Dense(256, name=name, activation='relu')(inputs)
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
    net = concatenate([net, embed], axis=1)
    net = Dense(1, name=name, activation='sigmoid')(net)
    return net



def get_model():
    logging.info("Building the model for " + function)
    input_seq = Input(shape=(MAXLEN,), dtype='int32', name='seq')
    input_embed = Input(shape=(256,), dtype='float32', name='embed')
    input_ipros = Input(shape=(len(interpros),), dtype='float32', name='ipros')
    net = get_function_node(get_node_name(function), input_seq, input_embed)
    model = Model([input_seq, input_embed], net)
    model.summary()
    logging.info('Compiling the model')
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy')
    logging.info('Compilation finished')
    return model


def train_model(train_df, valid_df, model_file, batch_size, epochs):
    start_time = time.time()
    logging.info("Training data size: %d" % len(train_df))
    logging.info("Validation data size: %d" % len(valid_df))

    checkpointer = ModelCheckpoint(
        filepath=model_file + '.h5',
        verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=16, verbose=1)
    logger = CSVLogger(model_file + '.csv')

    logging.info('Starting training the model')

    train_generator = DFGenerator(batch_size)
    train_generator.fit(train_df)
    valid_generator = DFGenerator(batch_size)
    valid_generator.fit(valid_df)

    valid_steps = int(math.ceil(len(valid_df) / batch_size))
    train_steps = int(math.ceil(len(train_df) / batch_size))

    model = get_model()
    
    model.summary()
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        max_queue_size=batch_size,
        workers=12,
        callbacks=[logger, checkpointer, earlystopper])

    
def test_model(test_df, model_file, batch_size):
    generator = DFGenerator(batch_size)
    generator.fit(test_df)
    steps = int(math.ceil(len(test_df) / batch_size))
    model = load_model(model_file + '.h5')
    logging.info('Compiling model')
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    logging.info('Evaluating model')
    loss = model.evaluate_generator(generator, steps=steps)
    logging.info('Test loss %f' % loss)
    
    logging.info('Predicting')
    generator.reset()
    preds = model.predict_generator(generator, steps=steps)
    test_labels = np.zeros((len(test_df), 1), dtype=np.int32)
    for i, row in enumerate(test_df.itertuples()):
        if function in row.functions:
            test_labels[i, 0] = 1
    roc_auc = compute_roc(preds, test_labels)
    logging.info('ROC AUC: \t %f ' % (roc_auc, ))
    f, p, r, t, preds_max = compute_performance(preds, test_labels)
    logging.info('Fmax measure: \t %f %f %f %f' % (f, p, r, t))
    report = classification_report(test_labels.flatten(), preds_max.flatten())
    logging.info(report)
    f = open(model_file + '.rpt', 'w')
    f.write('ROC AUC: \t %f \n' % (roc_auc, ))
    f.write(report)
    f.close()
    # mcc = compute_mcc(preds_max, test_labels)
    # logging.info('MCC: \t %f ' % (mcc, ))
    # print('%.3f & %.3f & %.3f & %.3f & %.3f' % (
    #     f, p, r, roc_auc, mcc))



def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(preds, labels):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def compute_performance(preds, labels):
    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        if tp == 0:
            continue
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            t_max = threshold
            predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max



if __name__ == '__main__':
    main()
