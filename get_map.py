#!/bin/python
# Create a bijection between base32 triples and LANGUAGE.

from string import ascii_lowercase
from itertools import product
from scipy.spatial.distance import pdist
import editdistance
import numpy as np
import pdb
import sys

# --- loading and prep --- #
def get_base32():
    """
    Generate triples of base32 characters.
    """
    alphabet = ascii_lowercase
    base32_chars = alphabet + '234567'
    base32_triples = [''.join(x) for x in product(base32_chars, repeat=3)]
    return base32_triples

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
    for line in open(path, 'r'):
        sl = line.strip('\n').split(' ')
        word = sl[0].capitalize()
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

def base32_distances(base32_triples):
    """
    Get pairwise Levenshtein distances.
    This takes a little while (~10 minutes on my laptop).
    """
    N = len(base32_triples)
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
            dij = editdistance.eval(base32_triples[i], base32_triples[j])
            d[i, j] = dij
            d[j, i] = dij
    print ''
    return dij

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
def random_map(triples, vocabulary):
    """
    Totally random map, totally unconstrained, totally boring.
    """
    forward_mapping = dict(zip(triples, vocabulary))
    backward_mapping = dict(zip(vocabulary, triples))
    return forward_mapping, backward_mapping

def diverse_map(triples, vocabulary, vectors):
    """
    Map which aims to map pairs of similar base32 triples to pairs of dissimilar
    language words.
    """
    N = len(triples)
    A = base32_distances(triples)
    B = pdist(vectors)
    ordering = find_ordering(A, B)
    forward_mapping, backward_mapping = dict(), dict()
    for i in xrange(N):
        triple = triples[i]
        word = vocabulary[ordering[i]]
        forward_mapping[triple] = word
        backward_mapping[word] = triple
    return forward_mapping, backward_mapping

def get_map(triples, vocabulary, vectors=None, mapping='random'):
    """
    Prep and get a map.
    """
    N = len(triples)
    W = len(vocabulary)
    if W < N:
        print 'ERROR: Not enough words.'
        return False
    if W > N:
        print 'There are', W, 'elements in the vocabulary and only', N,
        print 'triples: subsetting.'
        vocabulary_subset = list(np.random.choice(vocabulary, N))
        vocabulary = vocabulary_subset
    if mapping == 'random':
        print 'Using random map.'
        forward_mapping, backward_mapping = random_map(triples, vocabulary)
    elif mapping == 'diverse':
        print 'Using diverse map.'
        if vectors is None:
            print 'ERROR: diverse map requires vectors.'
            return False
        forward_mapping, backward_mapping = diverse_map(triples, vocabulary, vectors)
    else:
        print 'ERROR: Not implemented :('
    return forward_mapping, backward_mapping

# --- translation --- #
def translate(onion_address, forward_mapping):
    """
    Translate from base32 into LANGUAGE.
    """
    # this is the hack to get 6 words, sorry world
    onion_address += 'on'
    triples = [onion_address[i*3:(i+1)*3] for i in xrange(6)]
    triples_mapped = [forward_mapping[x] for x in triples]
    return ' '.join(triples_mapped)
