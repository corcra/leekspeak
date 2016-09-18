#!/usr/bin/env python
# 
# author:   Stephanie Hyland (@corcra)
# date:     18/9/2016
#
# Purely functions to help maximise the difference between the different
# pairwise distance matrices.

from itertools import product
import numpy as np
import pdb
import sys

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

def find_ordering(A, B, eps=0.1):
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

def count_violations(A, B, cA=0.1, cB=0.1):
    """
    Find the number of times the pairwise similarity of the same position in
    both distance measures is belong some threshold. (cA, cB)
    """
    A_violations = A < cA
    B_violations = B < cB
    simultaneous_violations = A_violations & B_violations
    return np.sum(simultaneous_violations)

