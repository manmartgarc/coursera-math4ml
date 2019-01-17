# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 00:38:53 2019

@author:    manma
@purpose:   This script has some functions for homework the homework
            on Course 1
"""

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
    return round(sum([i ** 2 for i in x]) ** 0.5, 1)


def change_basis(v, a, b):
    """
    Change the basis of v with respect to a, b.
    """

    # check for orthogonality of a, b
    if dot(a, b) != 0:
        raise ValueError('a & b are not orthogonal')

    x = round(dot(v, a) / mag(a) ** 2, 1)
    y = round(dot(v, b) / mag(b) ** 2, 1)
    return [x, y]
