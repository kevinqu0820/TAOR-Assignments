# -*- coding: utf-8 -*-

"""SA for TSP

We apply simulated annealing to the travelling salesman problem (TSP).
Let `n` be the number of cities. We encode a solution as a numpy array
containing a permutation of 0, 1, ...., n - 1. For example, an array
`x = np.arange(n)` corresponds to a solution `0 - 1 - 2 - ... - (n - 1) - 0`.
"""

import collections
import os

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
    # The implementation of simulated annealing...

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

    # Question 4
    # Run simulated annealing and plot the result...


if __name__ == "__main__":
    import doctest

    n_failuers, _ = doctest.testmod(
        optionflags=doctest.ELLIPSIS + doctest.NORMALIZE_WHITESPACE
    )
    if n_failuers > 0:
        raise ValueError(f"failed {n_failuers} tests")

    main()
