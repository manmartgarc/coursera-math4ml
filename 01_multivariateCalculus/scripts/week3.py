# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:52:43 2019

@author:    manma
@project:   Mathematics for Machine Learning.
@purpose:   Week 3 neural networks programming assignment.
"""
import numpy as np

# %%===========================================================================
# # First load the worksheet dependencies.
# # Here is the activation function and its derivative.
# =============================================================================


def sigma(z):
    return 1 / (1 + np.exp(z))


def d_sigma(z):
    return np.cosh(z / 2) ** (-2) / 4


def reset_network(n1=6, n2=7, random=np.random):
    """
    This function initialises the network with it's structure,
    it also resets any training already done.
    """
    global W1, W2, W3, b1, b2, b3
    W1 = random.randn(n1, 1) / 2
    W2 = random.randn(n2, n1) / 2
    W3 = random.randn(2, n2) / 2
    b1 = random.randn(n1, 1) / 2
    b2 = random.randn(n2, 1) / 2
    b3 = random.randn(2, 1) / 2


def network_function(a0):
    """
    This function feeds forward each activation to the next layer.
    It returns all weighted sums and activations.
    """
    z1 = W1 @ a0 + b1
    a1 = sigma(z1)
    z2 = W2 @ a1 + b2
    a2 = sigma(z2)
    z3 = W3 @ a2 + b3
    a3 = sigma(z3)
    return a0, z1, a1, z2, a2, z3, a3


def cost(x, y):
    """
    This is the cost function of a neural network with respect
    to a training set.
    """
    return np.linalg.norm(network_function(x)[-1] - y)**2 / x.size


# %%===========================================================================
# Backpropagation
# =============================================================================


def J_W3(x, y):
    """
    Jacobian for the third layer weights.
    There is no need to edit this function.
    """
    # First get all the activations and weighted sums at each
    # layer of the network.
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    # We'll use the variable J to store parts of our result as
    # we go along, updating it in each line.
    # Firstly, we calculate dC/da3, using the expressions above.
    J = 2 * (a3 - y)
    # Next multiply the result we've calculated by the
    # derivative of sigma, evaluated at z3.
    J = J * d_sigma(z3)
    # Then we take the dot product (along the axis that holds the
    # training examples) with the final partial derivative,
    # i.e. dz3/dW3 = a2
    # and divide by the number of training examples, for the
    # average over all training examples.
    J = J @ a2.T / x.size
    # Finally return the result out of the function.
    return J


def J_b3(x, y):
    """
    Jacobian for the third layer biases
    """
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


def J_W2(x, y):
    """
    Jacobian for the second layer weights
    """
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    # the next two lines implement da3/da2, first σ' and then W3.
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    # then the final lines are the same as in J_W3 but with
    # the layer number bumped down.
    J = J * d_sigma(z2)
    J = J @ a1.T / x.size
    return J


def J_b2(x, y):
    """
    # As previously, fill in all the incomplete lines.
    # ===YOU SHOULD EDIT THIS FUNCTION===
    """
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


def J_W1(x, y):
    """
    # GRADED FUNCTION
    # Fill in all incomplete lines.
    # ===YOU SHOULD EDIT THIS FUNCTION===
    """
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    J = J * d_sigma(z1)
    J = J @ a0.T / x.size
    return J


def J_b1(x, y):
    """
    # Fill in all incomplete lines.
    # ===YOU SHOULD EDIT THIS FUNCTION===
    """
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = (J.T @ W2).T
    J = J * d_sigma(z1)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J
