#!/usr/bin/env python


import pandas as pd
import numpy as np
from utils import EXP_CODES, get_gene_ontology
import os
import requests
from aaindex import is_ok
import gzip


DATA_ROOT = 'data/swissexp/'
def to_pickle():
    prots = set()
    proteins = list()
    accessions = list()
    sequences = list()
    with open(DATA_ROOT + 'uniprot_sprot.tab') as f:
        for line in f:
            items = line.strip().split('\t')
            ids = items[0].split('|')
            proteins.append(ids[2])
            prots.add(ids[2])
            accessions.append(ids[1])
            sequences.append(items[1])

    # with open('data/cafa3/tremble_data.tab') as f:
    #     for line in f:
    #         items = line.strip().split('\t')
    #         if items[0] not in prots:
    #             prots.add(items[0])
    #             proteins.append(items[0])
    #             accessions.append(items[1])
    #             sequences.append(items[2])

    # with open('data/cafa3/uniprot_trembl.tab') as f:
    #     for line in f:
    #         items = line.strip().split('\t')
    #         if items[1] not in prots:
    #             proteins.append(items[1])
    #             accessions.append(items[0])
    #             sequences.append(items[2])
    seq_df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'sequences': sequences
    })
    annots_dict = dict()
    with open(DATA_ROOT + 'swissprot.tab') as f:
        for line in f:
            items = line.strip().split('\t')
            accessions = items[1].split(';')
            annots_dict[items[0]] = set(items[2:])

    # with open('data/cafa3/tremble_data.tab') as f:
    #     for line in f:
    #         items = line.strip().split('\t')
    #         prot_id = items[0]
    #         if prot_id in annots_dict:
    #             annots_dict[prot_id] |= set(items[3].split('; '))
    #         else:
    #             annots_dict[prot_id] = set(items[3].split('; '))
    # goa_df = pd.read_pickle('data/cafa3/goa_annots.pkl')
    # for i, row in goa_df.iterrows():
    #     prot_id = row['proteins']
    #     if prot_id in annots_dict:
    #         annots_dict[prot_id] |= set(row['annots'])
    #     else:
    #         annots_dict[prot_id] = set(row['annots'])
    proteins = list()
    annots = list()
    for prot, gos in annots_dict.items():
        annots.append(list(gos))
        proteins.append(prot)
    annots_df = pd.DataFrame({
        'proteins': proteins,
        'annots': annots
    })
    df = pd.merge(seq_df, annots_df, on='proteins')
    print(len(df))
    df.to_pickle(DATA_ROOT + 'swissprot.pkl')


def goa_pickle():
    goa = dict()

    prots = dict()
    with open('data/uni_uni.dat') as f:
        for line in f:
            it = line.strip().split('\t')
            prots[it[0]] = it[1]

    with open('data/cafa3/goa_all.gaf') as f:
        for line in f:
            it = line.strip().split('\t')
            acc = it[1]
            code = it[6]
            go_id = it[4]
            if acc not in goa:
                goa[acc] = set()
            goa[acc].add(go_id + '|' + code)
    accessions = list()
    annots = list()

    proteins = list()

    for access, gos in goa.items():
        if access in prots:
            accessions.append(access)
            proteins.append(prots[access])
            annots.append(list(gos))
        else:
            r = requests.get('http://www.uniprot.org/uniprot/' + access + '.fasta')
            it = r.text.split('|')
            if len(it) > 1 and it[1] in prots:
                print(access, it[1], prots[it[1]])
                access = it[1]
                accessions.append(access)
                proteins.append(prots[access])
                annots.append(list(gos))

    df = pd.DataFrame({'accessions': accessions, 'annots': annots, 'proteins': proteins})
    print(len(df))
    df.to_pickle('data/cafa3/goa_annots.pkl')


def filter_exp():
    df = pd.read_pickle(DATA_ROOT + 'swissprot.pkl')
    exp_codes = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC'])
    index = list()
    for i, row in df.iterrows():
        ok = False
        for go_id in row['annots']:
            code = go_id.split('|')[1]
            if code in exp_codes:
                ok = True
                break
        if ok and is_ok(row['sequences']):
            index.append(i)
    df = df.loc[index]
    print(len(df))
    df.to_pickle(DATA_ROOT + 'swissprot_exp.pkl')


def string_uni():
    mapping = dict()
    with open('data/string2uni.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            mapping[it[0].upper()] = it[1]

    with open('data/string_idmapping.dat') as f:
        for line in f:
            it = line.strip().split('\t')
            mapping[it[2].upper()] = it[0]

    with open('data/uniprot-string.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            mapping[it[1].upper()[:-1]] = it[0]

    deep_map = dict()

    with gzip.open('data/graph.mapping.out.gz') as f:
        for line in f:
            it = line.strip().split('\t')
            deep_map[it[1]] = it[0]

    embeds = dict()
    with gzip.open('data/graph_deep.out.gz') as f:
        next(f)
        next(f)
        for line in f:
            it = line.strip().split()
            st_id = deep_map[it[0]].upper()
            if st_id in mapping:
                ac_id = mapping[st_id]
                embeds[ac_id] = np.array(
                    list(map(float, it[1:])), dtype='float32')

    df = pd.DataFrame({
        'accessions': list(embeds.keys()),
        'embeddings': list(embeds.values())})
    print(len(df))
    df.to_pickle('data/graph_new_embeddings.pkl')


def idmapping():
    access = list()
    proteins = list()
    with open('data/uni_uni.dat') as f:
        for l in f:
            it = l.strip().split('\t')
            access.append(it[0])
            proteins.append(it[1])
    print(len(access), len(proteins))
    df = pd.DataFrame({'accessions': access, 'proteins': proteins})

    access = list()
    string = list()
    with open('data/string_idmapping.dat') as f:
        for l in f:
            it = l.strip().split('\t')
            access.append(it[0])
            string.append(it[2])
    st_df = pd.DataFrame({'accessions': access, 'string': string})
    df = pd.merge(df, st_df, on='accessions', how='left')

    access = list()
    genes = list()
    with open('data/uni2ncbi.tab') as f:
        for l in f:
            it = l.strip().split('\t')
            access.append(it[0])
            genes.append(it[1])

    gene_df = pd.DataFrame({'accessions': access, 'genes': genes})
    df = pd.merge(df, gene_df, on='accessions', how='left')

    access = list()
    orgs = list()
    with open('data/uni2org.tab') as f:
        for l in f:
            it = l.strip().split('\t')
            access.append(it[0])
            orgs.append(it[1])

    org_df = pd.DataFrame({'accessions': access, 'orgs': orgs})
    df = pd.merge(df, org_df, on='accessions', how='left')
    df.to_pickle('data/idmapping.pkl')


def idmapping_org(org_id):
    df = pd.read_pickle('data/idmapping.pkl')
    df = df.loc[df['orgs'] == org_id]
    df.to_pickle('data/idmapping.' + org_id + '.pkl')


def predictions(org_id):
    preds = dict()
    with open('data/cafa3/done/model1/cbrcborg_1_' + org_id + '.txt') as f:
        next(f)
        next(f)
        next(f)
        for line in f:
            if line.strip() == 'END':
                continue
            it = line.strip().split('\t')
            score = float(it[2])
            if score < 0.35:
                continue
            target_id = it[0]
            go_id = it[1]
            if target_id not in preds:
                preds[target_id] = list()
            preds[target_id].append(go_id)
    targets = list()
    predicts = list()
    for t, p in preds.items():
        targets.append(t)
        predicts.append(p)
    df = pd.DataFrame({'targets': targets, 'predictions': predicts})
    tar_df = pd.read_pickle('data/cafa3/targets.pkl')
    tar_df = tar_df.loc[tar_df['orgs'] == org_id]
    id_df = pd.read_pickle('data/idmapping.' + org_id + '.pkl')
    tar_df = pd.merge(tar_df, id_df, on='proteins', how='left')
    df = pd.merge(df, tar_df, on='targets', how='left')
    df.to_pickle('data/human_predictions.pkl')
    testing = set()
    with open('data/cafa3/human_test.tab') as f:
        for line in f:
            testing.add(line.strip())
    with open('data/human_predictions.tab', 'w') as f:
        for i, row in df.iterrows():
            if not isinstance(row['string'], str):
                continue
            if row['proteins'] in testing:
                f.write(row['string'])
                for go_id in row['predictions']:
                    f.write('\t' + go_id)
                f.write('\n')


def human_go_annotations():
    go = get_gene_ontology()
    annots = {}
    df = pd.read_pickle('data/cafa3/swissprot_exp.pkl')
    for i, row in df.iterrows():
        acc = row['accessions']
        gos = set()
        for go_id in row['annots']:
            go_id = go_id.split('|')
            if go_id[1] in EXP_CODES and go_id[0] in go:
                gos.add(go_id[0])
        if len(gos) > 0:
            annots[acc] = gos
    id_df = pd.read_pickle('data/idmapping.9606.pkl')
    st_ids = dict()
    for i, row in id_df.iterrows():
        if isinstance(row['string'], str):
            st_ids[row['accessions']] = row['string']
    with open('data/human_annotations.tab', 'w') as f:
        for acc, gos in annots.items():
            if acc in st_ids:
                f.write(st_ids[acc])
                for go_id in gos:
                    f.write('\t' + go_id)
                f.write('\n')


def filter_goa():
    ROOT = 'data/goa/'
    files = os.listdir(ROOT)
    files = [filename for filename in files if not filename.startswith('gp2protein')]
    print(files)
    fw = open(ROOT + 'goa_all.gaf', 'w')
    for filename in files:
        with open(ROOT + filename) as f:
            for line in f:
                if not line.startswith('UniProtKB'):
                    continue
                it = line.strip().split('\t')
                if it[6] not in EXP_CODES:
                    continue
                fw.write(line)
    fw.close()


def gp2protein(org):
    mapping = {}
    with open('data/goa/gp2protein.' + org) as f:
        for line in f:
            if line.startswith('!'):
                continue
            it = line.strip().split('\t')
            if len(it) < 2:
                continue
            # ind = it[0].find(':')
            # prot_id = it[0][ind + 1:]
            # if org == 'tair':
            #     uni_ids = it[1].split('|')
            # else:
            #     uni_ids = it[1].split(';')
            prot_id = it[1]
            uni_ids = [it[0]]
            for uni_id in uni_ids:
                # if uni_id.startswith('UniProtKB:'):
                #     uni_id = uni_id[10:]
                if prot_id not in mapping:
                    mapping[prot_id] = list()
                mapping[prot_id].append(uni_id)

    fw = open('data/goa/goa_' + org + '.gaf', 'w')
    with open('data/goa/gene_association.' + org) as f:
        for line in f:
            if line.startswith('!'):
                continue
            it = list(line.strip().split('\t'))
            if it[6] not in EXP_CODES:
                continue
            if it[1] in mapping:
                for uni_id in mapping[it[1]]:
                    fw.write('UniProtKB\t' + uni_id)
                    for i in range(2, len(it)):
                        fw.write('\t' + it[i])
                    fw.write('\n')


def download_prots():
    df = pd.read_pickle('data/cafa3/tremble_prots.pkl')
    i = 0
    with open('data/cafa3/uniprot_trembl.dat', 'w') as f:
        for acc in df['accessions']:
            r = requests.get('http://www.uniprot.org/uniprot/' + acc + '.fasta')
            f.write(r.text + '\n')
            print(i)
            i += 1


def merge_trembl():
    seqs = {}
    with open('data/cafa3/tremble_sequences.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            seqs[it[0]] = it[1]
    w = open('data/cafa3/tremble_data.tab', 'w')
    with open('data/cafa3/tremble.tab') as f:
        for line in f:
            it = line.strip().split('\t')
            w.write(it[0] + '\t' + it[1] + '\t' + seqs[it[0]] + '\t' + it[2])
            for i in range(3, len(it)):
                w.write('; ' + it[i])
            w.write('\n')
    w.close()


def main():
    string_uni()
    # human_go_annotations()
    # predictions('9606')
    # to_pickle()
    # filter_exp()
    # goa_pickle()
    # download_prots()
    # merge_trembl()


if __name__ == '__main__':
    main()
