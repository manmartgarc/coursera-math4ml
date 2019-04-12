# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:46:58 2019

@author:    Manuel Martinez
@project:   Mathematics for Machine Learning.
@purpose:   Week 2 PCA programming assignment
"""

# PACKAGE: DO NOT EDIT THIS LINE
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy

import sklearn
# from ipywidgets import interact
# from load_data import load_mnist


def distance(x0, x1):
    """Compute distance between two vectors x0, x1 using the dot product"""
    d_vec = x0 - x1
    distance = np.sqrt(d_vec @ d_vec)
    return distance


def angle(x0, x1):
    """Compute the angle between two vectors x0, x1 using the dot product"""
    dot = x0 @ x1
    l1 = np.sqrt(x0 @ x0)
    l2 = np.sqrt(x1 @ x1)
    angle = np.arccos(dot / (l1 * l2))
    return angle


# find most similar
# np.argmin([distance(images[0], images[i]) for i in range(1, len(images))]) + 1

# GRADED FUNCTION: DO NOT EDIT THIS LINE
def most_similar_image():
    """Find the index of the digit, among all MNIST digits
       that is the second-closest to the first image in the dataset
       (the first image is closest to itself trivially).
       Your answer should be a single integer.
    """
    index = 61
    return index


# GRADED FUNCTION: DO NOT EDIT THIS LINE
def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)

    Returns
    --------
    distance_matrix: matrix of shape (N, M), each entry distance_matrix[i,j]
    is the distance between ith row of X and the jth row of Y
    (we use the dot product to compute the distance).
    """
    if len(Y.shape) == 1:
        Y = Y.reshape(1, -1)
    N, D = X.shape
    M, D = Y.shape
    xydiff = X[:, :, None] - Y[:, :, None].T
    distance_matrix = np.sqrt((xydiff * xydiff).sum(1))
    return distance_matrix


# GRADED FUNCTION: DO NOT EDIT THIS LINE
def KNN(k, X, y, x):
    """K nearest neighbors
    k: number of nearest neighbors
    X: training input locations
    y: training labels
    x: test input
    """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    N, D = X.shape
    num_classes = len(np.unique(y))
    dist = pairwise_distance_matrix(X, x)
    if dist.shape[1] == 1:
        dist = dist.ravel()

    # Next we make the predictions
    ypred = np.zeros(num_classes)
    # find the labels of the k nearest neighbors
    classes = y[np.argsort(dist)][:k]
    for c in classes:
        ypred[c] += 1

    return np.argmax(ypred)
