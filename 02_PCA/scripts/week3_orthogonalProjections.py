# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 23:32:08 2019

@author:    Manuel Martinez
@project:   Mathematics for Machine Learning.
@purpose:   Week 3 programming assignment on orthogonal projections
"""
# =============================================================================
# initialize environment
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as np_test
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
image_shape = (64, 64)
dataset = fetch_olivetti_faces('./')
faces = dataset.data

# %%===========================================================================
# 1. Orthogonal Projections
# =============================================================================


def test_property_projection_matrix(P):
    """Test if the projection matrix satisfies certain properties.
    In particular, we should have P @ P = P, and P = P^T
    """
    np_test.assert_almost_equal(P, P @ P)
    np_test.assert_almost_equal(P, P.T)


def test_property_projection(x, p):
    """Test orthogonality of x and its projection p."""
    np_test.assert_almost_equal(p.T @ (p-x), 0)


# GRADED FUNCTION: DO NOT EDIT THIS LINE

# Projection 1d

# ===YOU SHOULD EDIT THIS FUNCTION===
def projection_matrix_1d(b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        b: ndarray of dimension (D, 1), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    D, _ = b.shape
    # Edit the code below to compute a projection matrix of shape (D,D)
    P = (b @ b.T) / np.linalg.norm(b) ** 2
    return P


# ===YOU SHOULD EDIT THIS FUNCTION===
def project_1d(x, b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        x: the vector to be projected
        b: ndarray of dimension (D, 1), the basis for the subspace

    Returns:
        y: ndarray of shape (D, 1) projection of x in space spanned by b
    """
    P = projection_matrix_1d(b)
    p = P @ x
    return p


# Projection onto a general (higher-dimensional) subspace
# ===YOU SHOULD EDIT THIS FUNCTION===
def projection_matrix_general(B):
    """Compute the projection matrix onto the space spanned by the
    columns of `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    P = B @ np.linalg.inv(B.T @ B) @ B.T
    return P


# ===YOU SHOULD EDIT THIS FUNCTION===
def project_general(x, B):
    """Compute the projection matrix onto the space spanned by the
    columns of `B`
    Args:
        x: ndarray of dimension (D, 1), the vector to be projected
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        p: projection of x onto the subspac spanned by the columns of B;
        size (D, 1)
    """
    P = projection_matrix_general(B)
    p = P @ x
    return p


# %%===========================================================================
# 2. Eigenfaces
# =============================================================================
# plt.figure(figsize=(10, 10))
# plt.imshow(np.hstack(faces[:5].reshape(5, 64, 64)), cmap='gray')

# for numerical reasons we normalize the dataset
mean = faces.mean(axis=0)
std = faces.std(axis=0)
faces_normalized = (faces - mean) / std

# we use the first 50 basis vectors --- you should play around with this.
B = np.load('eigenfaces.npy')[:50]
print("the eigenfaces have shape {}".format(B.shape))

plt.figure(figsize=(10, 10))
plt.imshow(np.hstack(B[:5].reshape(-1, 64, 64)), cmap='gray')


# EDIT THIS FUNCTION
def show_face_face_reconstruction(i):
    original_face = faces_normalized[i].reshape(64, 64)
    # reshape the data we loaded in variable `B`
    # so that we have a matrix representing the basis.
    B_basis = B.reshape(B.shape[0], -1).T
    face_reconstruction = project_general(faces_normalized[i],
                                          B_basis).reshape(64, 64)
    plt.figure()
    plt.imshow(np.hstack([original_face, face_reconstruction]), cmap='gray')
    plt.show()


# %%===========================================================================
# 3. Least squares regression
# =============================================================================
x = np.linspace(0, 10, num=50)
theta = 2


def f(x):
    # we use the same random seed so we get deterministic output
    random = np.random.RandomState(42)
    # our observations are corrupted by some noise,
    # so that we do not get (x,y) on a line
    return theta * x + random.normal(scale=1.0, size=len(x))


y = f(x)
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')

# size N x 1
X = x.reshape(-1, 1)
# size N x 1
Y = y.reshape(-1, 1)

# maximum likelihood estimator
theta_hat = np.linalg.solve(X.T @ X, X.T @ Y)
# or
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y

fig, ax = plt.subplots()
ax.scatter(x, y)
xx = [0, 10]
yy = [0, 10 * theta_hat[0, 0]]
ax.plot(xx, yy, 'red', alpha=0.5)
ax.set(xlabel='x', ylabel='y')
print("theta = %f" % theta)
print("theta_hat = %f" % theta_hat)

# %%
N = np.arange(2, 10000, step=10)
# Your code comes here, which calculates \hat{\theta} for different
# dataset sizes.
y = f(x)
theta_error = np.ones(N.shape)

for i, n in enumerate(N):
    x = np.linspace(0, 10, num=n)
    y = f(x)
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    theta_hat = theta_hat = np.linalg.solve(X.T @ X, X.T @ Y)
    theta_error[i] = 2 - theta_hat

plt.plot(theta_error)
plt.xlabel("dataset size")
plt.ylabel("parameter error")
