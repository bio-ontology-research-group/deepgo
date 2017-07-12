AALETTER = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AANUM = len(AALETTER)
AAINDEX = dict()
for i in range(len(AALETTER)):
    AAINDEX[AALETTER[i]] = i
INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', 'X', '*'])
MAXLEN = 1002


def is_ok(seq):
    if len(seq) > MAXLEN:
        return False
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True

