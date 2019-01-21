# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:05:33 2019

@author: manma
"""

# %%
# PACKAGE
# Run this cell once first to load the dependancies.
# There is no need to submit this cell.
import numpy as np
from numpy.linalg import norm, inv
from numpy import transpose
# from readonly.bearNecessities import *
from week4_grahamSchmidt import gsBasis

# %%
# GRADED FUNCTION
# This is the cell you should edit and submit.

# In this function, you will return the transformation matrix T,
# having built it out of an orthonormal basis set E that you create from
# Bear's Basis and a transformation matrix in the mirror's coordinates TE.


def build_reflection_matrix(bearBasis):
    """
    # The parameter bearBasis is a 2×2 matrix that is passed to the function.
    # Use the gsBasis function on bearBasis to get the mirror's orthonormal
    basis.
    """
    E = gsBasis(bearBasis)
    # Write a matrix in component form that perform's the mirror's
    # reflection in the mirror's basis.
    # Recall, the mirror operates by negating the last component of a vector.
    # Replace a,b,c,d with appropriate values
    TE = np.array([[1, 0],
                   [0, -1]])
    # Combine the matrices E and TE to produce your transformation matrix.
    T = E @ TE @ E.T
    # Finally, we return the result. There is no need to change this line.
    return T
