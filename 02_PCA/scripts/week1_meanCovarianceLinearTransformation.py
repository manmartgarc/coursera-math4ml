# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:29:47 2019

@author:    manma
@project:   Mathematics for Machine Learning.
@purpose:   Week 1 PCA programming assignment
"""
import numpy as np


# GRADED FUNCTION: DO NOT EDIT THIS LINE
def mean_naive(X):
    "Compute the mean for a dataset X nby iterating over the data points"
    # X is of size (D,N) where D is the dimensionality
    # and N the number of data points
    D, N = X.shape
    mean = np.zeros((D, 1))
    # Edit the code; iterate over the dataset and compute the mean vector.
    for d in range(D):
        mean[d] = sum(X[d, :]) / N
    return mean


def cov_naive(X):
    """Compute the covariance for a dataset of size (D,N)
    where D is the dimension and N is the number of data points"""
    D, N = X.shape
    # Edit the code below to compute the covariance matrix
    # by iterating over the dataset.
    covariance = np.zeros((D, D))
    # Update covariance
    for d in range(D):
        d_mean = sum(X[d, :]) / N
        d_diffs = [x - d_mean for x in X[d, :]]
        for j in range(D):
            j_mean = sum(X[j, :]) / N
            j_diffs = [x - j_mean for x in X[j, :]]
            prods = [a * b for (a, b) in zip(d_diffs, j_diffs)]
            cov_dj = sum(prods) / N
            covariance[d, j] = cov_dj
    return covariance


def mean(X):
    """
    Compute the mean for a dataset of size (D,N) where D is the dimension
    and N is the number of data points
    """
    # given a dataset of size (D, N), the mean should be an array of size (D,1)
    # you can use np.mean, but pay close attention to the
    # shape of the mean vector you are returning.
    D, N = X.shape
    # Edit the code to compute a (D,1) array `mean` for the mean of dataset.
    mean = X.mean(axis=1).reshape(D, 1)
    return mean


def cov(X):
    """
    Compute the covariance for a dataset
    """
    # X is of size (D,N)
    # It is possible to vectorize our code for computing the
    # covariance with matrix multiplications,
    # i.e., we do not need to explicitly
    # iterate over the entire dataset as looping in Python tends to be slow
    # We challenge you to give a vectorized implementation without
    # using np.cov, but if you choose to use np.cov,
    # be sure to pass in bias=True.
    D, N = X.shape
    # Edit the code to compute the covariance matrix
    covariance_matrix = np.zeros((D, D))
    mean = X.mean(axis=1).reshape(D, 1)
    U = X - mean
    covariance_matrix = U @ U.T / N
    return covariance_matrix


# GRADED FUNCTION: DO NOT EDIT THIS LINE
def affine_mean(mean, A, b):
    """Compute the mean after affine transformation
    Args:
        x: ndarray, the mean vector
        A, b: affine transformation applied to x
    Returns:
        mean vector after affine transformation
    """
    # Edit the code below to compute the mean vector after
    # affine transformation
    affine_m = np.zeros(mean.shape)  # affine_m has shape (D, 1)
    affine_m = (A @ mean) + b
    return affine_m


def affine_covariance(S, A, b):
    """Compute the covariance matrix after affine transformation
    Args:
        S: ndarray, the covariance matrix
        A, b: affine transformation applied to each element in X
    Returns:
        covariance matrix after the transformation
    """
    # EDIT the code below to compute the covariance matrix
    # after affine transformation
    affine_cov = np.zeros(S.shape)  # affine_cov has shape (D, D)
    affine_cov = A @ S @ A.T
    return affine_cov
