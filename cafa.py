#!/usr/bin/env python
import sys
import os
import numpy as np
import pandas as pd
from aaindex import is_ok
import gzip as gz
from utils import EXP_CODES, get_gene_ontology, get_anchestors

MAXLEN = 1000


def get_fly_mapping():
    map1 = dict()
    with open('data/fly_uni.dat') as f:
        for line in f:
            it = line.strip().split('\t')
            map1[it[0]] = it[1]
    res = dict()
    with open('data/fly_idmapping.dat') as f:
        for line in f:
            it = line.strip().split('\t')
            if it[0] in map1:
                res[it[1]] = map1[it[0]]
    return res


def read_fasta(filename):
    data = list()
    c = 0
    with open(filename, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    data.append(seq)
                line = line[1:].split()[0]
                # line = line[1] + '\t' + line[2]
                seq = line + '\t'
            else:
                seq += line
        data.append(seq)
    print(c)
    return data


def get_annotations():
    w = open('data/cafa3/tremble.tab', 'w')
    with gz.open('data/uniprot_trembl.dat.gz', 'r') as f:
        prot_id = ''
        prot_ac = ''
        annots = list()
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '' and len(annots) > 0:
                    w.write(prot_id + '\t' + prot_ac)
                    for go_id in annots:
                        w.write('\t' + go_id)
                    w.write('\n')
                prot_id = items[1]
                annots = list()
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac = items[1]
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    if code in EXP_CODES:
                        annots.append(go_id + '|' + code)

        if len(annots) > 0:
            w.write(prot_id + '\t' + prot_ac)
            for go_id in annots:
                w.write('\t' + go_id)
            w.write('\n')
        w.close()


def get_sequences():
    prots = set()
    with open('data/cafa3/tremble.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            prots.add(it[0])
    w = open('data/cafa3/tremble_sequences.tab', 'w')
    with gz.open('data/uniprot_trembl.dat.gz', 'r') as f:
        prot_id = ''
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                prot_id = items[1]
            elif items[0] == 'SQ':
                if prot_id not in prots:
                    continue
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq
                w.write(prot_id + '\t' + seq + '\n')
        w.close()


def fasta2tabs():
    cafa_root = 'data/cafa3/CAFA3_targets/'
    data = list()
    # for dr in os.listdir(cafa_root):
    # if os.path.isdir(cafa_root + 'Targets/'):
    for fl in os.listdir('data/eshark/'):
        if fl.endswith('.fasta'):
            seqs = read_fasta('data/eshark/' + fl)
            data += seqs
    with open('data/eshark/targets.txt', 'w') as f:
        for line in data:
            f.write(line + '\n')


def sprot2tabs():
    data = read_fasta('data/cafa3/uniprot_trembl.fasta')
    with open('data/cafa3/uniprot_trembl.tab', 'w') as f:
        for line in data:
            f.write(line + '\n')


def cafa3():
    root = 'data/cafa3/CAFA3_training_data/'
    filename = root + 'uniprot_sprot_exp.fasta'
    data = read_fasta(filename)
    annots = dict()
    with open(root + 'uniprot_sprot_exp.txt') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[0] not in annots:
                annots[items[0]] = set()
            annots[items[0]].add(items[1])
    fl = open(root + 'uniprot_sprot.tab', 'w')
    for line in data:
        items = line.split('\t')
        if is_ok(items[1]) and items[0] in annots:
            fl.write(line + '\t')
            gos = list(annots[items[0]])
            fl.write(gos[0])
            for go_id in gos[1:]:
                fl.write('; ' + go_id)
            fl.write('\n')

def get_blast_mapping():
    mapping = {}
    with open('data/eshark/eshark.out') as f:
        for line in f:
            # if not line.startswith('evm.model'):
            #     continue
            it = line.strip().split()
            mapping[it[0]] = it[1]
    return mapping

def get_data():
    proteins = list()
    targets = list()
    orgs = list()
    ngrams = list()
    ngram_df = pd.read_pickle('data/eshark/ngrams.pkl')
    vocab = {}
    mapping = get_blast_mapping()
    for key, gram in enumerate(ngram_df['ngrams']):
        vocab[gram] = key + 1
    gram_len = len(ngram_df['ngrams'][0])
    print(('Gram length:', gram_len))
    print(('Vocabulary size:', len(vocab)))

    with open('data/eshark/targets.txt') as f:
        for line in f:
            it = line.strip().split('\t')
            seq = it[1]
            if is_ok(seq):
                # orgs.append(it[0])
                targets.append(it[0])
                if it[0] in mapping:
                    proteins.append(mapping[it[0]])
                else:
                    proteins.append('')
                grams = np.zeros((len(seq) - gram_len + 1, ), dtype='int32')
                for i in range(len(seq) - gram_len + 1):
                    grams[i] = vocab[seq[i: (i + gram_len)]]
                ngrams.append(grams)

    df = pd.DataFrame({
        'targets': targets,
        'accessions': proteins,
        'ngrams': ngrams})
    
    print((len(df)))
    embed_df = pd.read_pickle('data/graph_new_embeddings.pkl')

    df = pd.merge(df, embed_df, on='accessions', how='left')

    missing_rep = 0
    for i, row in df.iterrows():
        if not isinstance(row['embeddings'], np.ndarray):
            row['embeddings'] = np.zeros((256,), dtype='float32')
            missing_rep += 1
    print(missing_rep)

    df.to_pickle('data/eshark/targets.pkl')


def cafa2string():
    rep_prots = set()
    with open('data/uni_mapping.tab') as f:
        for line in f:
            items = line.strip().split('\t')
            rep_prots.add(items[0])
    c = 0
    with open('data/cafa3/targets.txt') as f:
        for line in f:
            items = line.strip().split('\t')
            if items[0] in rep_prots:
                c += 1
    print(c)


def get_real_annotations():
    go = get_gene_ontology()
    df = pd.read_pickle('data/cafa3/swissprot_exp.pkl')
    annots = {}
    for i, row in df.iterrows():
        go_set = set()
        for go_id in row['annots']:
            go_id = go_id.split('|')
            if go_id[0] in go and go_id[1] in EXP_CODES:
                go_set |= get_anchestors(go, go_id[0])
        annots[row['proteins']] = go_set
    return annots


def get_results(model):
    root = 'data/swissprot/done/'
    mf_df = pd.read_pickle(root + 'mf.pkl')
    cc_df = pd.read_pickle(root + 'cc.pkl')
    bp_df = pd.read_pickle(root + 'bp.pkl')
    targets = pd.read_pickle(root + 'targets.pkl')
    mf_preds = pd.read_pickle(root + model + '_preds_mf.pkl')
    mf_preds = mf_preds.rename(index=str, columns={"predictions": "mf"})
    cc_preds = pd.read_pickle(root + model + '_preds_cc.pkl')
    cc_preds = cc_preds.rename(index=str, columns={"predictions": "cc"})
    bp_preds = pd.read_pickle(root + model + '_preds_bp.pkl')
    bp_preds = bp_preds.rename(index=str, columns={"predictions": "bp"})
    df = pd.merge(targets, mf_preds, on='targets')
    df = pd.merge(df, cc_preds, on='targets')
    df = pd.merge(df, bp_preds, on='targets')
    mf = list(map(str, mf_df['functions'].values))
    cc = list(map(str, cc_df['functions'].values))
    bp = list(map(str, bp_df['functions'].values))
    taxons = set(df['orgs'].values)
    annots = get_real_annotations()
    for tax_id in taxons:
        res_df = df.loc[df['orgs'] == tax_id]
        results = {}
        for i, row in res_df.iterrows():
            prot_id = str(row['proteins'])
            target_id = str(row['targets'])
            if target_id not in results:
                results[target_id] = {}
            scores = np.round(row['mf'], 2)
            for j, go_id in enumerate(mf):
                score = scores[j]
                if score >= 0.01:
                    results[target_id][go_id] = score
            scores = np.round(row['cc'], 2)
            for j, go_id in enumerate(cc):
                score = scores[j]
                if score >= 0.01:
                    results[target_id][go_id] = score
            scores = np.round(row['bp'], 2)
            for j, go_id in enumerate(bp):
                score = scores[j]
                if score >= 0.01:
                    results[target_id][go_id] = score
            if prot_id in annots:
                for go_id in annots[prot_id]:
                    results[target_id][go_id] = 1.0

        with open(root + 'model3/' + 'cbrcborg_3_' + tax_id + '.txt', 'w') as f:
            f.write('AUTHOR CBRC_BORG\n')
            f.write('MODEL 3\n')
            f.write('KEYWORDS sequence properties, machine learning.\n')
            for target_id, annots in results.items():
                for go_id, score in annots.items():
                    sc = '%.2f' % score
                    f.write(target_id + '\t' + go_id + '\t' + sc + '\n')
            f.write('END\n')


def get_predictions():
    root = 'data/cafa3/'
    annots = {}
    preds = {}
    go = get_gene_ontology()
    mf = pd.read_pickle(root + 'mf.pkl')
    mf_df = pd.read_pickle(root + 'test-mf-preds.pkl')
    functions = mf['functions']
    for i, row in mf_df.iterrows():
        prot_id = row['proteins']
        if prot_id not in preds:
            preds[prot_id] = set()
        for i in range(len(functions)):
            if row['predictions'][i] == 1:
                preds[prot_id].add(functions[i])
        if prot_id not in annots:
            annots[prot_id] = row['gos']

    cc = pd.read_pickle(root + 'cc.pkl')
    cc_df = pd.read_pickle(root + 'test-cc-preds.pkl')
    functions = cc['functions']
    for i, row in cc_df.iterrows():
        prot_id = row['proteins']
        if prot_id not in preds:
            preds[prot_id] = set()
        for i in range(len(functions)):
            if row['predictions'][i] == 1:
                preds[prot_id].add(functions[i])
        if prot_id not in annots:
            annots[prot_id] = row['gos']

    bp = pd.read_pickle(root + 'bp.pkl')
    bp_df = pd.read_pickle(root + 'test-bp-preds.pkl')
    functions = bp['functions']
    for i, row in bp_df.iterrows():
        prot_id = row['proteins']
        if prot_id not in preds:
            preds[prot_id] = set()
        for i in range(len(functions)):
            if row['predictions'][i] == 1:
                preds[prot_id].add(functions[i])
        if prot_id not in annots:
            annots[prot_id] = row['gos']

    # Removing parent classes
    for prot_id in preds:
        go_set = preds[prot_id]
        gos = go_set.copy()
        for go_id in gos:
            anchestors = get_anchestors(go, go_id)
            anchestors.remove(go_id)
            go_set -= anchestors

    proteins = sorted(list(annots.keys()), key=lambda x: (
        x.split('_')[1], x.split('_')[0]))
    with open(root + 'test_predictions.tab', 'w') as f:
        for prot_id in proteins:
            f.write(prot_id)
            for go_id in preds[prot_id]:
                f.write('\t' + go_id)
            f.write('\n')

    with open(root + 'test_annotations.tab', 'w') as f:
        for prot_id in proteins:
            f.write(prot_id)
            for go_id in annots[prot_id]:
                if go_id in go:
                    f.write('\t' + go_id)
            f.write('\n')


def specific_predictions():
    root = 'data/cafa3/'
    go = get_gene_ontology()
    fw = open(root + 'test_predictions_specific.tab', 'w')
    with open(root + 'test_predictions.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            go_set = set(items[1:])
            gos = go_set.copy()
            for go_id in gos:
                anchestors = get_anchestors(go, go_id)
                anchestors.remove(go_id)
                go_set -= anchestors
            fw.write(items[0])
            for go_id in go_set:
                fw.write('\t' + go_id)
            fw.write('\n')
    fw.close()


def merged_annotations():
    root = 'data/cafa3/'
    preds = {}
    with open(root + 'test_predictions.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            preds[items[0]] = set(items[1:])
    fw = open(root + 'test_merged.tab', 'w')
    with open(root + 'test_annotations.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            gos = preds[items[0]] | set(items[1:])
            fw.write(items[0])
            for go_id in gos:
                fw.write('\t' + go_id)
            fw.write('\n')
    fw.close()


def compute_performance():
    root = 'data/cafa3/'
    preds = {}
    annots = {}
    go = get_gene_ontology()
    with open(root + 'test_predictions.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            preds[items[0]] = set(items[1:])
    with open(root + 'test_annotations.tab', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            annots[items[0]] = set()
            for go_id in items[1:]:
                if go_id in go:
                    annots[items[0]] |= get_anchestors(go, go_id)

    total = 0
    p = 0.0
    r = 0.0
    f = 0.0
    for prot, pred_annots in preds.items():
        real_annots = annots[prot]
        if len(real_annots) == 0:
            continue
        tp = len(real_annots.intersection(pred_annots))
        fp = len(pred_annots - real_annots)
        fn = len(real_annots - pred_annots)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        total += 1
        if tp != 0:
            precision = tp / (1.0 * (tp + fp))
            recall = tp / (1.0 * (tp + fn))
            p += precision
            r += recall
            f += 2 * precision * recall / (precision + recall)
    print((f / total, p / total, r / total))


def main(*args, **kwargs):
    # specific_predictions()
    # get_predictions()
    # merged_annotations()
    # compute_performance()
    # get_results('model_seq')
    get_data()
    # cafa3()
    # fasta2tabs()
    # cafa2string()
    # get_annotations()
    # get_sequences()
    # sprot2tabs()


if __name__ == '__main__':
    main(*sys.argv)
