#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:22:42 2023

@author: Kevin
"""

import numpy as np

def consistency_index(A):
    """

    Parameters
    ----------
    A : ndarray
        Pairwise comparison matrix, must be square.

    Raises
    ------
    ValueError
        If the input matrix is not square.

    Returns
    -------
    weights.real: list
        Relative importance weights.
    CI : scalar
        Consistency index.
        
    """
    
    # Check if the matrix is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square")
        
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Find the maximum eigenvalue and its corresponding eigenvector
    max_eigenvalue_index = np.argmax(eigenvalues)
    principal_eigenvector = eigenvectors[:, max_eigenvalue_index]

    # Normalize the principal eigenvector to get the relative importance weights
    weights = principal_eigenvector / np.sum(principal_eigenvector)

    # Compute the consistency index (CI)
    n = A.shape[0]
    max_eigenvalue = eigenvalues[max_eigenvalue_index].real
    CI = (max_eigenvalue - n) / (n - 1)

    return weights.real, CI


def generate_random_matrix(n):
    """
    Parameters
    ----------
    n : scalar
        Dimensions of matrix generated.

    Returns
    -------
    matrix : ndarray
        n x n random matrix.

    """
    matrix = np.identity(n)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Generate random value from Saaty's scale (1, 2, 3, 4, 5, 6, 7, 8, 9)
            value = np.random.choice(np.arange(1, 10))
            matrix[i, j] = value
            matrix[j, i] = 1 / value
    return matrix

def random_index(n, num_matrices=500, max_value=9):
    """

    Parameters
    ----------
    n : scalar
        Dimensions of random pairwise comparison matrices.
    num_matrices : scalar, optional
        Number of random pairwise comparison matrices used to compute the 
        random index. The default is 500.
    max_value : scalar, optional
        Maximum of random values in each random pairwise comparison matrix. 
        The default is 9.

    Returns
    -------
    RI : scalar
        Random index for dimension n pairwise comparison matrices.

    """
    # Initialise sum of CI for the random matrices computed so far
    total_CI = 0

    # Generate num_matrices random pairwise comparison matrices and compute their CI values
    for _ in range(num_matrices):
        random_matrix = generate_random_matrix(n)
        _, CI = consistency_index(random_matrix)
        total_CI += CI

    # Calculate the average CI value to obtain the random index (RI)
    RI = total_CI / num_matrices
    return RI
    

A = np.array([[1,5,2,4],[1/5,1,1/2,1/2],[1/2,2,1,2],[1/4,2,1/2,1]])
w,CI = consistency_index(A)

for n in range(2,11):
    print(random_index(n))