import numpy
from collections import deque
from sklearn.preprocessing import OneHotEncoder
from aaindex import AALETTER

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'


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

encoder = OneHotEncoder()


def init_encoder():
    data = list()
    for l in AALETTER:
        data.append([ord(l)])
    encoder.fit(data)

init_encoder()


def encode_seq_one_hot(seq, maxlen=500):
    data = list()
    for l in seq:
        data.append([ord(l)])
    data = encoder.transform(data).toarray()
    data = list(data)
    data = data[:maxlen]
    while (len(data) < maxlen):
        data.append([0] * 20)
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
    rng_state = numpy.random.get_state()
    for arg in args:
        if seed is not None:
            numpy.random.seed(seed)
        else:
            numpy.random.set_state(rng_state)
        numpy.random.shuffle(arg)
