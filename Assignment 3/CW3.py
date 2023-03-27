#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:22:42 2023

@author: Kevin
"""

import numpy as np
from matplotlib import pyplot as plt

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
            # Generate random value from Saaty's scale (1, 1/2, 1/3, ..., 1/9, 2, 3, ..., 9)
            value = np.random.choice(np.concatenate([np.arange(1, 10), 1 / np.arange(1, 10)]))
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
    

# Generate RI values for different dimensions
dimensions = np.arange(2,11)
RI = np.zeros(9)
for i in range(len(dimensions)):
    RI[i] = random_index(dimensions[i])
    
# Plot computed RI values against values in class
RI_lecture = np.array([0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.51])
plt.plot(dimensions, RI_lecture, label = 'Lecture results')
plt.plot(dimensions, RI, label = 'Our results')
plt.legend()
plt.xlabel('Dimensions')
plt.ylabel('RI')
plt.savefig('RI.pdf', dpi = 450)
plt.show()

# Define pairwise comparison matrices for the laptops in each criterion
A_Pr = np.array([[1, 2, 6], [1/2, 1, 3], [1/6, 1/3, 1] ])
A_Pe = np.array([[1, 1/2, 1/4], [2, 1, 1/2], [4, 2, 1]])
A_BL = np.array([[1, 1/3, 2/3], [3, 1, 2], [3/2, 1/2, 1]])
A_De = np.array([[1, 5/2, 1/2], [2/5, 1, 1/5], [2, 5, 1]])

# Compute the weights and CI values for each pairwise comparison matrix
w_Pr,CI_Pr= consistency_index(A_Pr)
w_Pe,CI_Pe = consistency_index(A_Pe)
w_BL,CI_BL = consistency_index(A_BL)
w_De,CI_De = consistency_index(A_De)

# Concatenate weights to form the scores of each laptop in each critoerion
scores = np.array([w_Pr, w_Pe, w_BL, w_De])
print(scores)

# Define pairwise comparison matrix for the objectives
A = np.array([[1, 1, 2, 5], [1, 1, 3, 5], [1/2, 1/3, 1, 5/3], [1/5, 1/5, 3/5, 1]])

# Compute and output weights and CI vaue of A
w,CI = consistency_index(A)
print('w =', w)
print('CI =', CI)

# Compute  and output weighted scores of the laptops
Z = np.dot(w,scores)
print('Z =', Z)

# Compute and output RI (at n-4) and consistency ratio 
print('RI =',random_index(4))
print('CR =', CI/random_index(4))
    
    