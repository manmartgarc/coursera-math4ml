# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:15:48 2019

@author:    Manuel Martinez
@project:   Mathematics for Machine Learning.
@purpose:   Week 4 programming assignment on principal component analysis.
"""
# PACKAGE: DO NOT EDIT THIS CELL
import numpy as np
import timeit

# PACKAGE: DO NOT EDIT THIS CELL
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from load_data import load_mnist
MNIST = load_mnist()
images, labels = MNIST['data'], MNIST['target']

# %%===========================================================================
# main
# =============================================================================
plt.figure(figsize=(4, 4))
plt.imshow(images[0].reshape(28, 28), cmap='gray')

# %%===========================================================================
# 1. PCA
# =============================================================================


# GRADED FUNCTION: DO NOT EDIT THIS LINE
def normalize(X):
    """Normalize the given dataset X
    Args:
        X: ndarray, dataset

    Returns:
        (Xbar, mean, std): tuple of ndarray, Xbar is the normalized dataset
        with mean 0 and standard deviation 1; mean and std are the
        mean and standard deviation respectively.

    Note:
        You will encounter dimensions where the standard deviation is
        zero, for those when you do normalization the normalized data
        will be NaN. Handle this by setting using `std = 1` for those
        dimensions when doing normalization.
    """
    mu = X.mean(axis=0)
    std = np.std(X, axis=0)
    std_filled = std.copy()
    std_filled[std == 0] = 1
    Xbar = (X - mu) / std_filled
    return Xbar, mu, std


def eig(S):
    """Compute the eigenvalues and corresponding eigenvectors
        for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix

    Returns:
        (eigvals, eigvecs): ndarray, the eigenvalues and eigenvectors

    Note:
        the eigenvals and eigenvecs should be sorted in descending
        order of the eigen values
    """
    eigvals, eigvecs = np.linalg.eig(S)
    desc_order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[desc_order]
    eigvecs = eigvecs[:, desc_order]
    return (eigvals, eigvecs)


def projection_matrix(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    P = B @ np.linalg.inv(B.T @ B) @ B.T
    return P


def PCA(X, num_components):
    """
    Args:
        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: ndarray of the reconstruction
        of X from the first `num_components` principal components.
    """
    N, D = X.shape
    S = X.T @ X / N
    _, B = eig(S)
    B = B[:, :num_components]
    P = projection_matrix(B)
    X_reconstruct = (P @ X.T).T
    return X_reconstruct


# %% Some preprocessing of the data
NUM_DATAPOINTS = 1000
X = (images.reshape(-1, 28 * 28)[:NUM_DATAPOINTS]) / 255.
Xbar, mu, std = normalize(X)

# %% take a look at how squared distance changes with n_dimensions
for num_component in range(1, 20):
    from sklearn.decomposition import PCA as SKPCA
    # We can compute a standard solution given by scikit-learn's
    # implementation of PCA
    pca = SKPCA(n_components=num_component, svd_solver='full')
    sklearn_reconst = pca.inverse_transform(pca.fit_transform(Xbar))
    reconst = PCA(Xbar, num_component)
    np.testing.assert_almost_equal(reconst, sklearn_reconst)
    print(np.square(reconst - sklearn_reconst).sum())


# %%
def mse(predict, actual):
    """Helper function for computing the mean squared error (MSE)"""
    return np.square(predict - actual).sum(axis=1).mean()


loss = []
reconstructions = []
# iterate over different number of principal components, and compute the MSE
for num_component in range(1, 100):
    reconst = PCA(Xbar, num_component)
    error = mse(reconst, Xbar)
    reconstructions.append(reconst)
#    print('n = {:d}, reconstruction_error = {:f}'.format(num_component, error))
    loss.append((num_component, error))

reconstructions = np.asarray(reconstructions)
# "unnormalize" the reconstructed image
reconstructions = reconstructions * std + mu
loss = np.asarray(loss)

# %%
fig, ax = plt.subplots()
ax.plot(loss[:, 0], loss[:, 1])
ax.axhline(100, linestyle='--', color='r', linewidth=2)
ax.xaxis.set_ticks(np.arange(1, 100, 5))
ax.set(xlabel='num_components', ylabel='MSE',
       title='MSE vs number of principal components')


# %%
# GRADED FUNCTION: DO NOT EDIT THIS LINE
# PCA for high dimensional datasets
def PCA_high_dim(X, n_components):
    """Compute PCA for small sample size but high-dimensional features.
    Args:
        X: ndarray of size (N, D), where D is the dimension of the sample,
           and N is the number of samples
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: (N, D) ndarray. the reconstruction
        of X from the first `num_components` pricipal components.
    """
    N, D = X.shape
    _, V = eig(X @ X.T)
    U = (X.T @ V)[:, :n_components]
    P = projection_matrix(U)
    X_reconstruct = (P @ X.T).T

    return X_reconstruct


# %%
np.testing.assert_almost_equal(PCA(Xbar, 2), PCA_high_dim(Xbar, 2))
print('A-OK')


# %%
def time(f, repeat=10):
    times = []
    for _ in range(repeat):
        start = timeit.default_timer()
        f()
        stop = timeit.default_timer()
        times.append(stop-start)
    return np.mean(times), np.std(times)


# %%
times_mm0 = []
times_mm1 = []

# iterate over datasets of different size
for datasetsize in np.arange(4, 784, step=20):
    XX = Xbar[:datasetsize] # select the first `datasetsize` samples in the dataset
    # record the running time for computing X.T @ X
    mu, sigma = time(lambda : XX.T @ XX)
    times_mm0.append((datasetsize, mu, sigma))

    # record the running time for computing X @ X.T
    mu, sigma = time(lambda : XX @ XX.T)
    times_mm1.append((datasetsize, mu, sigma))

times_mm0 = np.asarray(times_mm0)
times_mm1 = np.asarray(times_mm1)

fig, ax = plt.subplots()
ax.set(xlabel='size of dataset', ylabel='running time')
bar = ax.errorbar(times_mm0[:, 0], times_mm0[:, 1], times_mm0[:, 2], label="$X^T X$ (PCA)", linewidth=2)
ax.errorbar(times_mm1[:, 0], times_mm1[:, 1], times_mm1[:, 2], label="$X X^T$ (PCA_high_dim)", linewidth=2)
ax.legend();

%time Xbar.T @ Xbar
%time Xbar @ Xbar.T
# Put this here so that our output does not show result of computing `Xbar @ Xbar.T`
pass

# %%
times0 = []
times1 = []

# iterate over datasets of different size
for datasetsize in np.arange(4, 784, step=100):
    XX = Xbar[:datasetsize]
    npc = 2
    mu, sigma = time(lambda : PCA(XX, npc), repeat=10)
    times0.append((datasetsize, mu, sigma))

    mu, sigma = time(lambda : PCA_high_dim(XX, npc), repeat=10)
    times1.append((datasetsize, mu, sigma))

times0 = np.asarray(times0)
times1 = np.asarray(times1)

fig, ax = plt.subplots()
ax.set(xlabel='number of datapoints', ylabel='run time')
ax.errorbar(times0[:, 0], times0[:, 1], times0[:, 2], label="PCA", linewidth=2)
ax.errorbar(times1[:, 0], times1[:, 1], times1[:, 2], label="PCA_high_dim", linewidth=2)
ax.legend();

%time PCA(Xbar, 2)
%time PCA_high_dim(Xbar, 2)
pass