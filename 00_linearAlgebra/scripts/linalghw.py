# -*- coding: utf-8 -*-I
"""
Created on Thu Jan 17 00:38:53 2019

@author:    manma
@purpose:   This script has some functions for homework the homework
            on Course 1
"""
import itertools
# %% Week 2


def dot(a, b):
    """
    Take the inner (dot) product of vectors [a, b]
    """
    return round(sum([a[i] * b[i] for i in range(len(a))]), 1)


def mag(x):
    """
    Returns the magnitude of vector x
    """
    return sum([i ** 2 for i in x]) ** 0.5


def change_basis(*args, v):
    """
    Change the basis of v with respect to n vectors.
    """
    # check for orthogonality of args
    x = [dot(a, b) for a, b in itertools.combinations(args, 2)]
    if min(x) != 0:
        raise ValueError('one of the pairs of basis vectors '
                         'are not orthogonal.')

    # calculate scalar projections
    comp_vi = [round(dot(v, arg) / mag(arg) ** 2, 1) for arg in args]

    # calculate vector projections
    proj_vi = [[elem * i for elem in v] for i in comp_vi]

    prompt1 = 'scalars are [{}, {}]'.format(*comp_vi)
    prompt2 = 'vectors are [{}, {}]'.format(*proj_vi)

    return print('\n'.join([prompt1, prompt2]))
