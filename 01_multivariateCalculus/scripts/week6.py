# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:45:32 2019

@author:    manma
@project:   Mathematics for Machine Learning.
@purpose:   Week 6 Fitting the distribution of heights data.
"""
import numpy as np


def f(x, mu, sig):
    """
    This function is the Gaussian function.
    """
    return np.exp(-(x - mu) ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi) / sig


# Next up, the derivative with respect to μ.
# If you wish, you may want to express this as f(x, mu, sig)
# multiplied by chain rule terms.
# === COMPLETE THIS FUNCTION ===


def dfdmu(x, mu, sig):
    return f(x, mu, sig) * (x - mu) / sig ** 2


# Finally in this cell, the derivative with respect to σ.
# === COMPLETE THIS FUNCTION ===


def dfdsig(x, mu, sig):
    return f(x, mu, sig) * (((x - mu) ** 2 / sig ** 3) - 1 / sig)


def steepest_step(x, y, mu, sig, aggression):
    # Replace the ??? with the second element of the Jacobian.
    J = np.array([-2 * (y - f(x, mu, sig)) @ dfdmu(x, mu, sig),
                  -2 * (y - f(x, mu, sig)) @ dfdsig(x, mu, sig)])
    step = -J * aggression
    return step
