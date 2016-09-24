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

    Arguments:
        vocabulary      list of words (strings)
        vectors         list of vectors corresponding to vocabulary
        wordlist        str, list, or array, if str: path for .npy of words
        N               int: number of words we want vectors for

    Returns:
        vocabulary_subset       list of words (len N)
        vectors_subset          ndarray of vectors corresponding to words (shape[0] = N)
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
    vectors_subset = vectors[keep_indices]
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

def distance_lookup_table():
    """
    Pairwise character similarity lookup table.
   
    Characters in base32:
        base32_chars = alphabet + '234567'

    Similarities:
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
    base32_chars = ascii_lowercase + '234567'
    n_chars = len(base32_chars)
    # start off all ones
    distance_lookup = np.ones(shape=(n_chars, n_chars))
    
    # manually input the above
    # 0.5 may not be th best for this

    # b ~ d
    distance_lookup[base32_chars.index('b'), base32_chars.index('d')] = 0.5
    distance_lookup[base32_chars.index('d'), base32_chars.index('b')] = 0.5
    # p ~ q
    distance_lookup[base32_chars.index('p'), base32_chars.index('q')] = 0.5
    distance_lookup[base32_chars.index('q'), base32_chars.index('p')] = 0.5
    # m ~ n
    distance_lookup[base32_chars.index('m'), base32_chars.index('n')] = 0.5
    distance_lookup[base32_chars.index('n'), base32_chars.index('m')] = 0.5
    # v ~ w
    distance_lookup[base32_chars.index('v'), base32_chars.index('w')] = 0.5
    distance_lookup[base32_chars.index('w'), base32_chars.index('v')] = 0.5
    # c ~ e
    distance_lookup[base32_chars.index('c'), base32_chars.index('e')] = 0.5
    distance_lookup[base32_chars.index('e'), base32_chars.index('c')] = 0.5
    # a ~ o
    distance_lookup[base32_chars.index('a'), base32_chars.index('o')] = 0.5
    distance_lookup[base32_chars.index('o'), base32_chars.index('a')] = 0.5

    # zeros along the diagonal
    for i in xrange(n_chars):
        distance_lookup[i, i] = 0
           
    # the horrible dict version (i'm sure there's a lovely zip way to do this but w/e)
    distance_lookup_dict = dict()
    for (i, a) in enumerate(base32_chars):
        for (j, b) in enumerate(base32_chars):
            distance_lookup_dict[(a, b)] = distance_lookup[i, j]

    return distance_lookup, distance_lookup_dict
        
def bespoke_distance(nmer1, nmer2, offset_kappa):
    """
    Hand-crafted distance function, probably not a real metric.
    Thinking about what is 'hard to differentiate', as human looking at strings
    Properties:
    - adjacent swaps are hard to detect
    - m ~ rn
    - see pairwise_character_distance

    """
    raise NotImplementedError
    d = 0
    # if the nmers are identical, don't need to do anything complicated
    if nmer1 == nmer2:
        return d
    # check exact correspondences (increment distances)
    for (a, b) in zip(nmer1, nmer2):
        d += pairwise_character_distance(a, b)      # this fn has bespoke distances
    # now check with an offset
    for (a, b) in zip(nmer1[1:]+nmer[0], nmer2):
        d += offset_kappa*pairwise_character_distance(a, b)      # this fn has bespoke distances
    # negative offset
    for (a, b) in zip(nmer[-1]+nmer1[:-1], nmer2):
        d += offset_kappa*pairwise_character_distance(a, b)      # this fn has bespoke distances
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

def visual_string_distance(a, b):
    """
    Gets the distance between two strings (a, b) using _visual_ features.

    Ideas:
    If the length of both is the same, equal to n:
        compare sliding window for swaps, remember which letters are involved
            in swaps, normalise by length of string
        compare letter by letter, using similarity table as for base32
            normalise by length of string 
                (so if 1 position differs with value 1, total dissim is 1/n)
            except for position 0, which is special and gets weighted more
            also exclude positions involved in swaps

    If the lengths are different, by d:
        if d > 3: return 1 (max dissimilar)
        else:
            slide shorter word along longer, see 'same length comparison'
            take care about position 0 while sliding
            penalise d depending on n (shorter word length)
    """
    raise NotImplementedError
    return 0.5

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
    B = squareform(pdist(vectors, 'cosine'))
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
