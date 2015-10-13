#!/bin/python
# Create a bijection between base32 nmers and LANGUAGE.

from string import ascii_lowercase
from itertools import product
from scipy.spatial.distance import pdist, squareform
import editdistance
import numpy as np
import pdb
import sys
import re
import gzip

NMER_SIZE = 3

# --- loading and prep --- #
def get_base32():
    """
    Generate nmers of base32 characters.
    """
    alphabet = ascii_lowercase
    base32_chars = alphabet + '234567'
    base32_nmers = [''.join(x) for x in product(base32_chars, repeat=NMER_SIZE)]
    return base32_nmers

def subset_language(vocabulary, vectors, wordlist, N=32768):
    """
    Subset the vocabulary/vectors to those in a wordlist.
    The wordlist is a list arranged in order of 'preference'.
    Note: we hope the vocabulary is contained in the wordlist,
    but it might not be. N is the number of words we require.
    If the wordlist contains fewer than N words, (but the vocabulary has >= N),
    we supplement the result from the vocabulary randomly.
    Also, we want to make sure the order of vocabulary is random (because some
    structure could negatively influence the optimisation procedure later).
    """
    keep_indices = []           # indices of vocabulary/vectors to keep
    added = 0
    if type(wordlist) == str:
        # load from path
        print 'Loading wordlist from', wordlist
        wordlist = np.loadtxt(wordlist, dtype=str)
    else:
        assert type(wordlist) == list or type(wordlist) == np.ndarray
    print 'Subsetting vocabulary.'
    for word in wordlist:
        print word
        if added == N:
            break
        try:
            word_index = vocabulary.index(word)
            keep_indices.append(word_index)
            added += 1
        except ValueError:
            continue
    print 'Acquired', len(keep_indices), 'words.'
    miss = N - len(keep_indices)
    if miss > 0:
        print 'Supplementing with', miss, 'random words.'
        for i in xrange(miss):
            random_index = np.random.choice(len(vocabulary), 1)
            while random_index in keep_indices:
                random_index = np.random.choice(len(vocabulary), 1)
            keep_indices.append(random_index)
    print 'Shuffling.'
    # shuffle
    np.random.shuffle(keep_indices)
    # populate new arrays
    print 'Populating subsetted arrays.'
    vectors_subset = np.array([vectors[i] for i in keep_indices])
    vocabulary_subset = [vocabulary[i] for i in keep_indices]
    return vocabulary_subset, vectors_subset

def get_language(path):
    """
    Get the LANGUAGE words, and vectors!
    Takes a path to a file like
        apple 0.3410 0.24 0.4114
        orange 0.613 3.414 0.512
    Outputs a list like     [apple, orange]
    and a np array like     [[0.3410, 0.24, 0.4114],
                             [0.613, 3.414, 0.512]]
    """
    print 'Loading language from', path
    vocabulary = []
    vectors = []
    if '.gz' in path:
        fi = gzip.open(path, 'rb')
    else:
        fi = open(path, 'r')
    for line in fi:
        if '\t' in line:
            sl = line.strip('\n').split('\t')
        else:
            sl = line.strip('\n').split(' ')
        word = re.sub('\x00', '', sl[0])
        vocabulary.append(word)
        if len(sl) > 1:
            vector = map(float, sl[1:])
        else:
            vector = np.random.normal(size=5)
        vectors.append(vector)
    vectors = np.array(vectors)
    W = len(vocabulary)
    print 'Loaded', W, 'words from', path
    return vocabulary, vectors

# --- distance metrics --- #

def bespoke_distance(nmer1, nmer2):
    """
    Hand-crafted distance function, probably not a real metric.
    Thinking about what is 'hard to differentiate', as human looking at strings
    Properties:
    - adjacent swaps are hard to detect
    - i ~ l (1 is not a problem as it does not exist in base32)
    - b ~ d
    - p ~ q
    - m ~ n
    - v ~ w
    - c ~ e
    - a ~ o
    Note: this is largely arbitrary from me, partially influenced by:
        http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3541865/table/t1-ptj3712663/
    """
    # I can already feel how slow this is going to be.
    # some stuff goes here etc zzzz
    d = abs(np.random.normal())
    return d

def base32_distances(base32_nmers, metric='levenshtein'):
    """
    Get pairwise distances (different metrics)
    This takes a little while
    """
    N = len(base32_nmers)
    total = N*(N-1.0)/2
    print 'Calculating', N*(N-1)/2, 'pairwise distances.'
    d = np.empty(shape=(N, N), dtype=np.float)
    n = 0
    for i in xrange(N):
        for j in xrange(i, N):
            n += 1
            if n%500000 == 0:
                sys.stdout.write('\r'+'%.4f' % (float(n*100)/total)+'%')
                sys.stdout.flush()
            if metric == 'levenshtein':
                dij = editdistance.eval(base32_nmers[i], base32_nmers[j])
            elif metric == 'bespoke':
                dij = bespoke_distance(base32_nmers[i], base32_nmers[j])
            else:
                raise NotImplementedError
            d[i, j] = dij
            d[j, i] = dij
    print ''
    return d

# --- some optimisation stuff --- #
def get_proposal(A, B):
    """
    Everyone loves MH.
    """
    n = A.shape[0]
    # get a pivot point
    inner_products = np.einsum('i...,...i', A, B)
    violated = np.random.choice(n, size=1, p=inner_products/np.sum(inner_products))[0]
    # get the rest
    v = np.array([x for (i, x) in enumerate(A[:, violated]) if not i == violated])
    phi = np.array([x for (i, x) in enumerate(B[:, violated]) if not i == violated])
    # now reorder... (argsort gives low to high, remember)
    v_order = np.argsort(v)
    phi_order = np.argsort(-phi)
    # want to move the highest phi to the lowest v
    ordering_subset = v_order[phi_order]
    ordering_subset[np.where(ordering_subset >= violated)] += 1
    # reinsert into ordering
    ordering = np.empty(shape=len(v)+1, dtype=np.int)
    ordering[:violated] = ordering_subset[:violated]
    ordering[violated] = violated
    ordering[(violated+1):] = ordering_subset[violated:]
    assert len(set(ordering)) == len(ordering)
    return ordering

def find_ordering(A, B, eps=0.00001):
    """
    Reorder the rows/columns of B to maximise its difference to A.
    ... possibly.
    Use Metropolis-Hastings for some reason.
    """
    assert A.shape[0] == A.shape[1]
    assert B.shape == A.shape
    diff = np.mean(abs(A - B))
    temperature = diff
    delta = 100
    cumulative_delta = 0
    print diff, delta
    accept, reject = 0, 0
    while abs(delta) > eps:
        # get proposal ordering
        proposal_ordering = get_proposal(A, B)
        proposal_B = B[proposal_ordering, :][:, proposal_ordering]
        proposal_diff = np.mean(abs(A - proposal_B))
        # accept with some probability
        proposal_delta = diff - proposal_diff
        prob = min(1, np.exp(-proposal_delta/temperature))
        if np.random.random() <= prob:
            accept += 1
            ordering = proposal_ordering
            diff = proposal_diff
            delta = proposal_delta
            cumulative_delta -= delta
            B = proposal_B
            print diff, -delta, cumulative_delta
            if accept%100 == 0:
                temperature /= 1.1
        else:
            reject += 1
            if reject%100 == 0:
                temperature *= 1.15
        temperature *= 0.99999
    acceptance_rate = float(accept)/(accept + reject)
    return ordering, acceptance_rate, temperature

# --- maps --- #
def random_map(nmers, vocabulary):
    """
    Totally random map, totally unconstrained, totally boring.
    """
    forward_mapping = dict(zip(nmers, vocabulary))
    backward_mapping = dict(zip(vocabulary, nmers))
    return forward_mapping, backward_mapping

def diverse_map(nmers, vocabulary, vectors):
    """
    Map which aims to map pairs of similar base32 nmers to pairs of dissimilar
    language words.
    """
    N = len(nmers)
    A = base32_distances(nmers)
    print A.shape
    B = squareform(pdist(vectors))
    print B.shape
    ordering = find_ordering(A, B)
    forward_mapping, backward_mapping = dict(), dict()
    for i in xrange(N):
        triple = nmers[i]
        word = vocabulary[ordering[i]]
        forward_mapping[triple] = word
        backward_mapping[word] = triple
    return forward_mapping, backward_mapping

def get_map(nmers, vocabulary, vectors=None, mapping='random'):
    """
    Prep and get a map.
    """
    N = len(nmers)
    W = len(vocabulary)
    if W < N:
        print 'ERROR: Not enough words.'
        return False
    if W > N:
        print 'There are', W, 'elements in the vocabulary and only', N,
        print 'nmers: subsetting.'
        vocabulary_subset = list(np.random.choice(vocabulary, N))
        vocabulary = vocabulary_subset
    if mapping == 'random':
        print 'Using random map.'
        forward_mapping, backward_mapping = random_map(nmers, vocabulary)
    elif mapping == 'diverse':
        print 'Using diverse map.'
        if vectors is None:
            print 'ERROR: diverse map requires vectors.'
            return False
        forward_mapping, backward_mapping = diverse_map(nmers, vocabulary, vectors)
    else:
        print 'ERROR: Not implemented :('
    # sanity check
    for (k, v) in forward_mapping.iteritems():
        if not backward_mapping[v] == k:
            print k, v
    return forward_mapping, backward_mapping
