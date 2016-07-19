import numpy as np
from collections import deque
from sklearn.preprocessing import OneHotEncoder
from aaindex import AAINDEX
import string

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
DIGITS = string.digits + string.letters
BASE = len(DIGITS)


def get_node_name(n):
    if n == 0:
        return '0'
    ret = ''
    while n > 0:
        ret = ret + DIGITS[n % BASE]
        n /= BASE
    return ret


class DataGenerator(object):

    def __init__(self, batch_size, num_outputs):
        self.batch_size = batch_size
        self.num_outputs = num_outputs

    def fit(self, inputs, targets):
        self.start = 0
        self.inputs = inputs
        self.targets = targets

    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < len(self.inputs):
            output = []
            labels = self.targets
            for i in range(self.num_outputs):
                output.append(
                    labels[i, self.start:(self.start + self.batch_size)])
            res_inputs = self.inputs[self.start:(self.start + self.batch_size)]
            self.start += self.batch_size
            return (res_inputs, output)
        else:
            self.reset()
            return self.next()


def get_gene_ontology(filename='go.obo'):
    # Reading Gene Ontology from OBO Formatted file
    go = dict()
    obj = None
    with open('data/' + filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                # elif l[0] == 'relationship':
                #     r = l[1].split(' ')
                #     if r[0] == 'part_of':
                #         obj['part_of'].append(r[1])
                #     elif r[0] == 'regulates':
                #         obj['regulates'].append(r[1])
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        go[obj['id']] = obj
    for go_id in go.keys():
        if go[go_id]['is_obsolete']:
            del go[go_id]
    for go_id, val in go.iteritems():
        if 'children' not in val:
            val['children'] = set()
        for g_id in val['is_a']:
            if g_id in go:
                if 'children' not in go[g_id]:
                    go[g_id]['children'] = set()
                go[g_id]['children'].add(go_id)
        # for g_id in val['part_of']:
        #     if g_id in go:
        #         if 'children' not in go[g_id]:
        #             go[g_id]['children'] = set()
        #         go[g_id]['children'].add(go_id)
        # for g_id in val['regulates']:
        #     if g_id in go:
        #         if 'children' not in go[g_id]:
        #             go[g_id]['children'] = set()
        #         go[g_id]['children'].add(go_id)
    # Rooting
    go['root'] = dict()
    go['root']['is_a'] = []
    go['root']['children'] = [
        BIOLOGICAL_PROCESS, MOLECULAR_FUNCTION, CELLULAR_COMPONENT]
    go[BIOLOGICAL_PROCESS]['is_a'] = ['root']
    go[MOLECULAR_FUNCTION]['is_a'] = ['root']
    go[CELLULAR_COMPONENT]['is_a'] = ['root']

    return go


def get_anchestors(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while(len(q) > 0):
        g_id = q.popleft()
        go_set.add(g_id)
        for parent_id in go[g_id]['is_a']:
            if parent_id in go:
                q.append(parent_id)
        # for parent_id in go[g_id]['part_of']:
        #     if parent_id in go:
        #         q.append(parent_id)
        # for parent_id in go[g_id]['regulates']:
        #     if parent_id in go:
        #         q.append(parent_id)
    return go_set


def get_parents(go, go_id):
    go_set = set()
    for parent_id in go[go_id]['is_a']:
        if parent_id in go:
            go_set.add(parent_id)
    # for parent_id in go[go_id]['part_of']:
    #     if parent_id in go:
    #         go_set.add(parent_id)
    # for parent_id in go[go_id]['regulates']:
    #     if parent_id in go:
    #         go_set.add(parent_id)
    return go_set


def get_go_sets(go, go_ids):
    go_set = set()
    q = deque()
    for go_id in go_ids:
        q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set


def get_go_set(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set


def get_disjoint_sets(go, go_id):
    sets = list()
    gos = list(go[go_id]['children'])
    for ch_id in gos:
        go_set = get_go_set(go, ch_id)
        sets.append(go_set)
    a = list()
    n = len(sets)
    for i in range(n):
        a.append([0] * n)
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i].intersection(sets[j]):
                a[i][j] = 1
                a[j][i] = 1
    for i in range(n):
        print a[i]
    used = set()

    def dfs(v):
        used.add(v)
        for i in range(n):
            if a[v][i] == 1 and i not in used:
                dfs(i)
    c = 0
    u = set()
    for i in range(n):
        if i not in used:
            dfs(i)
            group = list()
            for x in used:
                if x not in u:
                    group.append(gos[x])
                    u.add(x)
            print '------------'
            c += 1
            print len(get_go_sets(go, group))


encoder = OneHotEncoder()


def init_encoder():
    data = np.arange(1, 21).reshape(20, 1)
    encoder.fit(data)

init_encoder()


def encode_seq_one_hot(seq):
    res = np.zeros((len(seq), 20), dtype='float32')
    for i in range(len(seq)):
        res[i, :] = encoder.transform([[seq[i]]]).toarray()
    for i in range(len(seq)):
        print seq[i], res[i]
    return res


def encode_sequences(sequences, maxlen=1000):
    n = len(sequences)
    data = np.zeros((n, maxlen, 20), dtype='float32')
    for i in range(n):
        m = len(sequences[i])
        print m
        data[i, :m, :] = encode_seq_one_hot(sequences[i])
        break
    return data


def train_val_test_split(labels, data, split=0.8, batch_size=16):
    """This function is used to split the labels and data
    Input:
        labels - array of labels
        data - array of data
        split - percentage of the split, default=0.8\
    Return:
        Three tuples with labels and data
        (train_labels, train_data), (val_labels, val_data), (test_labels, test_data)
    """
    n = len(labels)
    train_n = int((n * split) / batch_size) * batch_size
    val_test_n = int((n - train_n) / 2)

    train_data = data[:train_n]
    train_labels = labels[:train_n]
    train = (train_labels, train_data)

    val_data = data[train_n:][0:val_test_n]
    val_labels = labels[train_n:][0:val_test_n]
    val = (val_labels, val_data)

    test_data = data[train_n:][val_test_n:]
    test_labels = labels[train_n:][val_test_n:]
    test = (test_labels, test_data)

    return (train, val, test)


def train_test_split(labels, data, split=0.8, batch_size=16):
    """This function is used to split the labels and data
    Input:
        labels - array of labels
        data - array of data
        split - percentage of the split, default=0.8\
    Return:
        Three tuples with labels and data
        (train_labels, train_data), (test_labels, test_data)
    """
    n = len(labels)
    train_n = int((n * split) / batch_size) * batch_size

    train_data = data[:train_n]
    train_labels = labels[:train_n]
    train = (train_labels, train_data)

    test_data = data[train_n:]
    test_labels = labels[train_n:]
    test = (test_labels, test_data)

    return (train, test)


def shuffle(*args, **kwargs):
    """
    Shuffle list of arrays with the same random state
    """
    seed = None
    if 'seed' in kwargs:
        seed = kwargs['seed']
    rng_state = np.random.get_state()
    for arg in args:
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.set_state(rng_state)
        np.random.shuffle(arg)


# get_disjoint_sets(get_gene_ontology(), CELLULAR_COMPONENT)

def get_statistics():
    go = get_gene_ontology('goslim_yeast.obo')
    print len(go)
    bp = get_go_set(go, BIOLOGICAL_PROCESS)
    mf = get_go_set(go, MOLECULAR_FUNCTION)
    cc = get_go_set(go, CELLULAR_COMPONENT)
    print len(bp), len(mf), len(cc)

# get_statistics()
