#!/bin/python
# Encode between onion address and language.

import re

def encode(onion_address, forward_mapping):
    """
    Translate from base32 into LANGUAGE.
    """
    # this is the hack to get 6 words, sorry world
    onion_address = re.sub('http:\/\/','', re.sub('.onion(\/)?$', '', onion_address)) + 'on'
    triples = [onion_address[i*3:(i+1)*3] for i in xrange(6)]
    triples_mapped = [forward_mapping[x] for x in triples]
    onion_language = ' '.join(triples_mapped)
    return onion_language

def decode(onion_language, backwards_mapping):
    """
    Translate from LANGUAGE into base32.
    """
    words = onion_language.split(' ')
    assert len(words) == 6
    words_mapped = [backwards_mapping[x] for x in words]
    onion_address = ''.join(words_mapped)[:-2]+'.onion'
    return onion_address
