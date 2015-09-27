#!/bin/python
# Create a bijection between base32 triples and LANGUAGE.

from string import ascii_lowercase
from itertools import product
import numpy as np

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

# --- maps --- #
def random_map(triples, vocabulary):
    """
    Totally random map, totally unconstrained, totally boring.
    """
    forward_mapping = dict(zip(triples, vocabulary))
    backward_mapping = dict(zip(vocabulary, triples))
    return forward_mapping, backward_mapping

def get_map(triples, vocabulary, vectors=None, mapping='random'):
    """
    Prep and get a map.
    """
    N = len(triples)
    W = len(vocabulary)
    if W < N:
        print 'ERROR! Not enough words.'
        return False
    if W > N:
        print 'There are', W, 'elements in the vocabulary and only', N,
        print 'triples: subsetting.'
        vocabulary_subset = list(np.random.choice(vocabulary, N))
        vocabulary = vocabulary_subset
    if mapping == 'random':
        print 'Using random map.'
        forward_mapping, backward_mapping = random_map(triples, vocabulary)
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
