# -*- coding: utf-8 -*-

"""SA for TSP

We apply simulated annealing to the travelling salesman problem (TSP).
Let `n` be the number of cities. We encode a solution as a numpy array
containing a permutation of 0, 1, ...., n - 1. For example, an array
`x = np.arange(n)` corresponds to a solution `0 - 1 - 2 - ... - (n - 1) - 0`.
"""

import collections
import os
import math
import random

import matplotlib.pyplot as plt

import numpy as np


def compute_total_cost(solution, distances):
    """Compute the total cost of a given solution

    Examples
    --------
    >>> solution = np.array([0, 1, 2, 3, 4, 5])
    >>> distances = np.array([
    ...    [0, 5, 3, 4, 2, 3],
    ...    [5, 0, 2, 8, 3, 9],
    ...    [3, 2, 0, 2, 5, 8],
    ...    [4, 8, 2, 0, 6, 9],
    ...    [2, 3, 5, 6, 0, 1],
    ...    [3, 9, 8, 9, 1, 0],
    ... ], dtype=float)
    >>> compute_total_cost(solution, distances)
    19.0

    Parameters
    ----------
    solution : ndarray
        1D array of shape (n_cities,) with `int` dtype representing
        the solution whose length is to be computed.
    distances : ndarray
        2D array of shape (n_cities, n_cities) with `float` dtype
        representing the distance matrix.

    Returns
    -------
    length : float
    """
    # Question 1
    # Append starting point to solution to form a route
    solution = np.append(solution, solution[0])
    # Initialise cost
    length = 0
    # Traverse the cities in solution in order and compute the distances 
    # between each city and the next city.
    for i in range(len(solution) - 1):
        length += distances[solution[i], solution[i+1]]
        
    return length
    
    


def run_greedy_heuristic(distances):
    """Run a greedy heuristic for TSP

    This runs a greedy heuristic for TSP and return a feasible solution.
    This starts at city 0 and creates a soltuion by finding the shortest
    cities greedily.

    Examples
    --------
    >>> distances = np.array([
    ...    [0, 5, 3, 4, 2, 3],
    ...    [5, 0, 2, 8, 3, 9],
    ...    [3, 2, 0, 2, 5, 8],
    ...    [4, 8, 2, 0, 6, 9],
    ...    [2, 3, 5, 6, 0, 1],
    ...    [3, 9, 8, 9, 1, 0],
    ... ], dtype=float)
    >>> run_greedy_heuristic(distances)
    array([0, 4, 5, 2, 1, 3])
    >>> compute_total_cost(run_greedy_heuristic(distances), distances)
    25.0

    Parameters
    ----------
    distances : ndarray
        2D array of shape (n_cities, n_cities) with `float` dtype
        representing the distance matrix.

    Returns
    -------
    solution : ndarray
        1D array of shape (n_cities,) with `int` dtype representing
        the solution obtained by the greedy heuristic.
    """
    # Question 2
    # Set starting city to 0
    c = 0
    # Initialise solution
    solution = np.array([c])
    # Initialise set of cities
    n = np.shape(distances)[0]
    S = set(range(1,n))
    # Initialise distances array to not obtain trivial solutions when computing
    # closest cities
    distances = distances + (np.max(distances) + 1)*np.eye(n)

    for i in range(1,n):
         # Update distance matrix to not revisit city c
         distances[:,c] = distances[:,c] + (np.max(distances))
         # Set new city to be the closest to c
         c_new = np.argmin(distances[c,:])
         # Remove new city from set S and append to route
         S.discard(c_new)
         solution = np.append(solution,c_new)
         # Set new city to be the next city
         c = c_new


        
    return solution
    


def sample_two_opt(solution):
    """Return a neighbour of a given solution based on two-opt

    This returns a neighbouring solution.

    Examples
    --------
    >>> solution = np.array([0, 1, 2, 3])
    >>> sample_two_opt(solution)  # doctest: +SKIP
    array([0, 2, 1, 3])

    Parameters
    ----------
    solution : ndarray
        1D array of shape (n_cities,) with `int` dtype representing
        the current solution.

    Returns
    -------
    new_solution : ndarray
        1D array of shape (n_cities,) with `int` dtype representing
        the sampled solution.
    """
    # Question 3
    n = len(solution)
    
    while True:
        # Sample 2 numbers from the solution space, making the smaller one i 
        # and the larger one j
        samples = np.random.permutation(n)[0:2]
        i = np.min(samples)
        j = np.max(samples)
        # if 1 < j-i < n-1, we accept them, otherwise we resample them
        if j - i > 1 and j - i < n - 1:
            break
    # Reverse the order of entries in solution between indices i and j  
    new_solution = np.concatenate((solution[0:i+1], solution[j:i:-1], solution[j+1:]))
    return new_solution

def run_simulated_annealing(
    initial_solution,
    objective,
    sample,
    n_epochs,
    temperature,
):
    """Run simulated annealing

    Parameters
    ----------
    solution : ndarray
        1D array of shape (n_cities,) with `int` dtype representing
        the initial solution.
    objective : callable
        Objective function of the following signature:
        (solution: ndarray of shape (n_cities,)) -> float.
    sample : callable
        A function to sample a neighbouring solution.
        This should have the following signature:
        (solution: ndarray of shape (n_cities,))
        -> ndarray of shape (n_cities,).
    n_epochs : int
        Number of epochs
    temperature : callable
        A function to compute the temperature.
        This should have the following signature:
        (epoch: int) -> float.

    Returns
    -------
    objective : float
        The objective value of the best solution found.
    solution : ndarray
        1D array of shape (n_cities,) with `int` dtype representing
        the best solution found.
    objective_list : list of float
        The objective values of the iterates
    """
    
    best_solution = None  # Store the best solution on this variable.
    best_objective = np.inf  # Store the obj. value of `best_solution` on this.
    objective_list = []  # List to store the objective values of the iterate.

    # Question 4
    x_list = []  # Store all the iterates
    x = initial_solution
    x_list.append(x)
    objective_list.append(objective(x))
    for epoch in range(n_epochs-1):
        #make a neighbourhood N(x), and select x' from N(x) randomly
        xp = sample(x)
        dE = objective(xp) - objective(x)
        p = min(1, math.exp(-(dE)/temperature(epoch)))
        if random.random() <= p:
            x = xp
            x_list.append(x)
            o = objective(x)
            objective_list.append(o)
            best_objective = min(objective_list)
            best_solution = [q for q, e in zip(x_list, objective_list) if e == best_objective][0]
    

    return SolverResult(best_objective, best_solution, objective_list)


SolverResult = collections.namedtuple(
    "SolverResult", "objective solution objective_list"
)

def main():
    """Run the main routine of this script"""
    distance_matrix_file_path = "distances.npy"

    with open(distance_matrix_file_path, "rb") as f:
        distances = np.load(f)

    # Run the greedy heuristic and obtain a solution.
    initial_solution = run_greedy_heuristic(distances)

    # Test the output.
    shape = (len(distances),)

    np.testing.assert_equal(type(initial_solution), np.ndarray)
    np.testing.assert_equal(initial_solution.shape, shape)

    # Test output of `sample_two_opt` as well.

    sampled_solution = sample_two_opt(initial_solution)

    np.testing.assert_equal(type(sampled_solution), np.ndarray)
    np.testing.assert_equal(sampled_solution.shape, shape)
    
    def objective(solution):
        y = compute_total_cost(solution, distances)
        return y

    # Question 4
    def temperature(k):
        #return 1/(0.1*k + 1)
        #return 100/(0.1*k + 1)
        return 0.1/(0.1*k + 1)
    
    for k in range(0, 20):
        
        simulated_annealing = run_simulated_annealing(initial_solution, objective, sample_two_opt, 1400, temperature)
        plt.plot(simulated_annealing[2], color = "blue", linewidth = 1)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.show()

if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")

    main()
