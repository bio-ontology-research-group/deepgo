#!/usr/bin/env python

"""
python predict.py
"""

import numpy as np
import pandas as pd
import click as ck
from keras.models import load_model
from keras.optimizers import RMSprop
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
    filter_specific)
from keras.preprocessing import sequence
from keras import backend as K
import sys
import time
import datetime
import logging
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
sys.setrecursionlimit(100000)

DATA_ROOT = 'data/eshark/'
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
    '--model-name',
    default='model',
    help='Name of the model')
def main(function, device, model_name):
    global FUNCTION
    FUNCTION = function
    global GO_ID
    GO_ID = FUNC_DICT[FUNCTION]
    global go
    go = get_gene_ontology('go.obo')
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
    # with tf.device('/' + device):
    #     model(model_name)
    # add_gos()
    to_csv()

def load_data():
    df = pd.read_pickle(DATA_ROOT + 'targets.pkl')

    def reshape(values):
        values = np.hstack(values).reshape(
            len(values), len(values[0]))
        return values

    def get_values(data_frame):
        ngrams = sequence.pad_sequences(
            data_frame['ngrams'].values, maxlen=MAXLEN)
        ngrams = reshape(ngrams)
        embeddings = reshape(data_frame['embeddings'].values)
        return (ngrams, embeddings)

    data = get_values(df)

    return data, df['targets'].values


def model(model_name):
    # set parameters:
    batch_size = 128
    nb_classes = len(functions)
    start_time = time.time()
    logging.info("Loading Data")
    data, targets = load_data()
    data_generator = DataGenerator(batch_size, nb_classes)
    data_generator.fit(data, None)

    logging.info("Data loaded in %d sec" % (time.time() - start_time))
    logging.info("Data size: %d" % len(data[0]))
    logging.info('Loading the model')
    model = load_model(
        DATA_ROOT + model_name + '_' + FUNCTION + '.h5')

    
    logging.info('Predicting')
    preds = model.predict_generator(
       data_generator, val_samples=len(data[0]))
    
    # incon = 0
    # for i in xrange(len(data)):
    #    for j in xrange(len(functions)):
    #         anchestors = get_anchestors(go, functions[j])
    #         for p_id in anchestors:
    #             if (p_id not in [GO_ID, functions[j]] and
    #                     preds[i, go_indexes[p_id]] < preds[i, j]):
    #                 incon += 1
    #                 preds[i, go_indexes[p_id]] = preds[i, j]
    # logging.info('Inconsistent predictions: %d' % incon)

    predictions = list()
    for i in range(len(targets)):
        predictions.append(preds[i])
    df = pd.DataFrame({
        'targets': targets,
        'predictions': predictions})
    print((len(df)))
    df.to_pickle(DATA_ROOT + model_name + '_preds_' + FUNCTION + '.pkl')
    logging.info('Done in %d sec' % (time.time() - start_time))


def add_gos():
    df = pd.read_pickle(DATA_ROOT + 'model_preds_' + FUNCTION + '.pkl')
    gos = list()
    threshold = 0.2
    for i, row in df.iterrows():
        preds = row['predictions']
        go_ids = list()
        for i in range(len(preds)):
            if preds[i] >= threshold:
                go_ids.append(functions[i])
        gos.append(filter_specific(go, go_ids))
    df['gos_' + FUNCTION] = gos
    print(df)
    df.to_pickle(DATA_ROOT + 'predictions_' + FUNCTION + '.pkl')


def to_csv():
    bp_df = pd.read_pickle(DATA_ROOT + 'predictions_bp.pkl').drop(
        'predictions', axis=1)
    mf_df = pd.read_pickle(DATA_ROOT + 'predictions_mf.pkl').drop(
        'predictions', axis=1)
    cc_df = pd.read_pickle(DATA_ROOT + 'predictions_cc.pkl').drop(
        'predictions', axis=1)
    df = bp_df.merge(mf_df, on='targets').merge(cc_df, on='targets')
    gos = list()
    go_names = list()
    for i, row in df.iterrows():
        go_ids = list()
        go_nms = list()
        for go_id in row['gos_bp']:
            go_ids.append('P:' + go_id)
            go_nms.append('P:' + go[go_id]['name'])
        for go_id in row['gos_mf']:
            go_ids.append('F:' + go_id)
            go_nms.append('F:' + go[go_id]['name'])
        for go_id in row['gos_cc']:
            go_ids.append('C:' + go_id)
            go_nms.append('C:' + go[go_id]['name'])
        gos.append(go_ids)
        go_names.append(go_nms)
    df = pd.DataFrame({
        'SeqName': df['targets'], 'GO_IDS': gos, 'GO_NAMES': go_names})
    print(df)
    dt = datetime.datetime.today().strftime('%Y%m%d')
    df.to_csv(
        DATA_ROOT + 'deepgo_%s.tsv' % (dt,),
        sep='\t', index=False, header=True,
        columns=['SeqName', 'GO_IDS', 'GO_NAMES'])
    
    
if __name__ == '__main__':
    main()
