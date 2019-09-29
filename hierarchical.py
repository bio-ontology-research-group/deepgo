#!/usr/bin/env python





import os
import sys
import scipy.sparse as sp
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import click as ck
from collections import deque
import math

from utils import (
    get_gene_ontology, get_go_set, get_parents,
    FUNC_DICT)
import tf_utils


@ck.command()
@ck.option(
    '--data-root',
    default='data/cafa3/',
    help='Path to folder with all necessary files')
@ck.option(
    '--go-filename',
    default='go.obo',
    help='GO filename in OBO Format')
@ck.option(
    '--go-domain',
    default='mf',
    help='Ontology domain (mf, bp, cc)')
@ck.option(
    '--split',
    default=0.7,
    help='Train test split')
def main(data_root, go_filename, go_domain, split):
    global DATA_ROOT
    DATA_ROOT = data_root
    global go
    go = get_gene_ontology()
    global FUNCTION
    FUNCTION = go_domain
    df = pd.read_pickle(DATA_ROOT + go_domain + '.pkl')
    global functions
    functions = list(df['functions'])
    global func_set
    func_set = set(functions)
    global GO_ID
    GO_ID = FUNC_DICT[FUNCTION]
    dataset = load_data(split=split)
    train_model(dataset)


def load_data(split=0.7):
    df = pd.read_pickle(DATA_ROOT + 'data-' + FUNCTION + '.pkl')
    n = len(df)
    index = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(index)
    train_n = int(n * split)
    valid_n = int(train_n * split)
    train_df = df.loc[index[:valid_n]]
    valid_df = df.loc[index[valid_n:train_n]]
    test_df = df.loc[index[train_n:]]

    def reshape(values):
        print(len(values), len(values[0]))
        values = np.hstack(values).reshape(
            len(values), len(values[0]))
        return values

    def pad_sequences(values, max_len=1000):
        for i in range(len(values)):
            padded = np.zeros((max_len,), dtype='int32')
            padded[:len(values[i])] = values[i][:]
            values[i] = padded
        return values

    train_input1, train_input2, train_labels = (
        reshape(pad_sequences(train_df['indexes'].values)),
        reshape(train_df['rep'].values),
        reshape(train_df['labels'].values))
    valid_input1, valid_input2, valid_labels = (
        reshape(pad_sequences(valid_df['indexes'].values)),
        reshape(valid_df['rep'].values),
        reshape(valid_df['labels'].values))
    test_input1, test_input2, test_labels = (
        reshape(pad_sequences(test_df['indexes'].values)),
        reshape(test_df['rep'].values),
        reshape(test_df['labels'].values))

    train = train_input1, train_input2, train_labels
    valid = valid_input1, valid_input2, valid_labels
    test = test_input1, test_input2, test_labels
    return train, valid, test


def train_model(dataset, batch_size=512, epochs=10):
    train, valid, test = dataset
    train_input1, train_input2, train_labels = train
    valid_input1, valid_input2, valid_labels = valid
    test_input1, test_input2, test_labels = test

    input1_length = train_input1.shape[1]
    input2_length = train_input2.shape[1]
    train_n = train_input1.shape[0]
    train_steps = int(math.ceil(train_n / batch_size))
    valid_n = valid_input1.shape[0]
    valid_steps = int(math.ceil(valid_n / batch_size))
    test_n = test_input1.shape[0]
    test_steps = int(math.ceil(test_n / batch_size))
    print('Training data size:', train_input1.shape, train_input2.shape)
    print('Validation data size:', valid_input1.shape, valid_input2.shape)
    print('Testing data size:', test_input1.shape, test_input2.shape)
    with tf.device('/gpu:1'):
        tf.reset_default_graph()
        placeholders = dict()
        placeholders['input1'] = tf.placeholder(
            tf.int32, shape=(None, input1_length))
        placeholders['input2'] = tf.placeholder(
            tf.float32, shape=(None, input2_length))
        for go_id in functions:
            placeholders[go_id] = tf.placeholder(
                tf.float32, shape=(None, 1))
        layers = model(placeholders)
        loss = 0
        for go_id in functions:
            loss += tf.nn.sigmoid_cross_entropy_with_logits(
                layers[go_id]['logits'], placeholders[go_id])
        loss = tf.reduce_mean(loss)
        trainer = tf.train.RMSPropOptimizer(learning_rate=0.01)
        update = trainer.minimize(loss)

        outputs = [loss]
        for i in range(len(functions)):
            go_id = functions[i]
            outputs.append(layers[go_id]['output'])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(epochs):
            print('Epoch %d/%d' % (epoch, epochs))
            sum_loss = 0.0
            with ck.progressbar(range(train_steps)) as bar:
                for step in bar:
                    offset = step * batch_size
                    batch_input1 = train_input1[offset:(offset + batch_size)]
                    batch_input2 = train_input2[offset:(offset + batch_size)]
                    feed_dict = {
                        placeholders['input1']: batch_input1,
                        placeholders['input2']: batch_input2,
                    }
                    for i in range(len(functions)):
                        go_id = functions[i]
                        feed_dict[placeholders[go_id]] = train_labels[offset:(offset + batch_size), i].astype('float32').reshape(-1, 1)
                    _, train_loss = sess.run(
                        [update, loss],
                        feed_dict=feed_dict)
                    sum_loss += train_loss
            print('Training loss:', sum_loss / train_steps)

            sum_loss = 0.0
            predictions = np.empty(
                (valid_n, len(functions)), dtype='float32')

            for step in range(valid_steps):
                offset = step * batch_size
                feed_dict = {
                    placeholders['input1']: valid_input1[offset:(offset + batch_size)],
                    placeholders['input2']: valid_input2[offset:(offset + batch_size)],
                }
                for i in range(len(functions)):
                    go_id = functions[i]
                    feed_dict[placeholders[go_id]] = valid_labels[offset:(offset + batch_size), i].reshape(-1, 1)

                results = sess.run(
                    outputs,
                    feed_dict=feed_dict)
                sum_loss += results[0]
                for i in range(len(functions)):
                    predictions[offset:(offset + batch_size), i] = results[i + 1].reshape(-1)
            print('Validation F1 score:', f_score(predictions, valid_labels))
            print('Validation loss:', sum_loss / valid_steps)

        sum_loss = 0.0
        predictions = np.empty(
            (test_n, len(functions)), dtype='float32')
        for step in range(test_steps):
            offset = step * batch_size
            feed_dict = {
                placeholders['input1']: test_input1[offset:(offset + batch_size)],
                placeholders['input2']: test_input2[offset:(offset + batch_size)],
            }
            for i in range(len(functions)):
                go_id = functions[i]
                feed_dict[placeholders[go_id]] = test_labels[offset:(offset + batch_size), i].reshape(-1, 1)

            results = sess.run(
                outputs,
                feed_dict=feed_dict)
            test_loss = results[0]
            sum_loss += test_loss
            for i in range(len(functions)):
                predictions[offset:(offset + batch_size), i] = results[i + 1].reshape(-1)
        print('Test F1 score:', f_score(predictions, test_labels))
        print('Test loss:', sum_loss / test_steps)


def f_score(preds, labels):
    preds = (preds > 0.5).astype('int32')
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
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            p += precision
            r += recall
            f += 2 * precision * recall / (precision + recall)
    return f / total, p / total, r / total


def merge(outputs):
    if len(outputs) == 1:
        return outputs[0]
    return tf_utils.concatenate(outputs, 1)


def gcn(inputs):
    graph = get_ppi_graph()
    n = int(inputs.get_shape()[1])
    weights = tf.Variable(
        tf.truncated_normal([n, len(graph)],
                            stddev=1.0 / math.sqrt(float(n))),
        name='gcn_weights')
    net = tf.matmul(inputs, weights)
    biases = tf.Variable(tf.zeros([len(graph)]),
                         name='gcn_biases')
    net = tf.nn.relu(tf.matmul(net, graph) + biases)
    return net


def features(inputs):
    # Embedding Layer
    vocabulary_size = 21
    embedding_size = 20
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    embed = tf.gather(embeddings, inputs)
    embed = tf.expand_dims(embed, -1)
    print('Embedding shape:', embed.get_shape())
    # 1D Convolutional Layer
    nb_filters = 32
    filter_length = 20
    filters = tf.Variable(tf.truncated_normal(
        [filter_length, filter_length, 1, nb_filters], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[nb_filters]))
    conv = tf.nn.conv2d(
        embed, filters, strides=[1, 1, 1, 1], padding='VALID')
    conv = tf.nn.relu(conv + biases)
    print('Conv:', conv.get_shape())
    # 1D Max Pooling Layer
    pool_length = 10
    pool_stride = 5
    pool = tf.nn.max_pool(
        conv,
        ksize=[1, pool_length, 1, 1],
        strides=[1, pool_stride, 1, 1],
        padding='VALID')
    pool = tf.squeeze(pool, [2])
    pool_shape = pool.get_shape()
    print('Pool', pool_shape)
    shape = int(pool_shape[1] * pool_shape[2])
    f = tf.reshape(pool, [-1, shape])
    return f


def model(placeholders, node_output_dim=128):
    q = deque()
    layers = dict()
    sequence_features = features(placeholders['input1'])
    net = merge([sequence_features, placeholders['input2']])
    layers[GO_ID] = {'net': slim.batch_norm(net)}
    for node_id in go[GO_ID]['children']:
        if node_id in func_set:
            q.append((node_id, node_output_dim))
    while len(q) > 0:
        node_id, output_dim = q.popleft()
        parents = get_parents(go, node_id)
        parent_outputs = list()
        for parent_id in parents:
            if parent_id in layers:
                parent_outputs.append(layers[parent_id]['net'])
        net = merge(parent_outputs)
        net = slim.fully_connected(net, output_dim)
        logits = slim.fully_connected(net, 1)
        output = tf.nn.sigmoid(logits)
        layers[node_id] = {'net': net, 'output': output, 'logits': logits}
        for n_id in go[node_id]['children']:
            if n_id in func_set:
                q.append((n_id, output_dim))
    return layers


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()


def get_ppi_graph():
    df = pd.read_pickle(DATA_ROOT + 'genes.pkl')
    genes = list(df['genes'])
    n = len(genes)
    index = dict()
    adj = np.zeros((n, n), dtype='float32')
    for i, gene in enumerate(genes):
        index[gene] = i
        adj[i, i] = 1.0
    with open(DATA_ROOT + 'interactions.human.txt') as f:
        next(f)
        for line in f:
            items = line.strip().split('\t')
            g1 = items[7]
            g2 = items[8]
            if g1 in index and g2 in index:
                adj[index[g1], index[g2]] = 1.0
                adj[index[g2], index[g1]] = 1.0
    return normalize_adj(adj)


if __name__ == '__main__':
    main()
