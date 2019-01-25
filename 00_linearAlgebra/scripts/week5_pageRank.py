# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 15:19:10 2019

@author:    manma
@project:   Mathematics for Machine Learning.

@purpose:   This code has the progamming assignment for Week 5.
"""
import numpy as np
import numpy.linalg as la
np.set_printoptions(suppress=True)


def generate_internet(n):
    c = np.full([n, n], np.arange(n))
    c = (abs(np.random.standard_cauchy([n, n]) / 2) > (np.abs(c - c.T) + 1))
    c = (c + 1e-10) / np.sum((c + 1e-10), axis=0)
    return c


def pageRank(linkMatrix, d):
    # get size of internet
    n = linkMatrix.shape[0]

    # initialize rank vector with equal probabilities
    r = np.ones(n) / n

    # initialize an n x n matrix filled with ones.
    J = np.ones(linkMatrix.shape)

    # initialize power iteration method for the link matrix.
    M = d * linkMatrix + (1 - d) / n * J

    # initialize rank vector and cached it
    lastR = r
    r = M @ r

    # power iteration
    while la.norm(lastR - r) > 0.01:
        lastR = r
        r = M @ r
    return r


# %% Use the following function to generate internets of different sizes.
L = generate_internet(1000)
pageRank(L, 1)

# %% calculating eigenvalues version
eVals, eVecs = la.eig(L)  # Gets the eigenvalues and vectors
order = np.absolute(eVals).argsort()[::-1]  # Orders them by their eigenvalues
eVals = eVals[order]
eVecs = eVecs[:, order]
r = eVecs[:, 0]
100 * np.real(r / np.sum(r))
