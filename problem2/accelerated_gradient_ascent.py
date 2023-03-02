# Standard Libraries
import os
import numpy as np
import cmath

def _k_partial(
    k : float,
    c : float,
    y : np.ndarray
) -> float:
    '''
    Computes the partial derivative with respect to k for the log-likelihood function
    for a set of observations and a given set of parameters.

    Parameters:
        k (float): First parameter of the log-likelihood function.
        c (float): Second parameter of the log-likelihood function.
        y (np.ndarray): All observations.

    Returns:
        partial_k (float):
            The value of the partial derivative with respect to k of the log-likelihood function
            for the given parameters and observations.
    '''
    partial_k = np.sum(1 / k - np.array([cmath.log(y_val**c + 1).real for y_val in y]))

    return partial_k

def _c_partial(
    k : float,
    c : float,
    y : np.ndarray
) -> float:
    '''
    Computes the partial derivative with respect to c for the log-likelihood function
    for a set of observations and a given set of parameters.

    Parameters:
        k (float): First parameter of the log-likelihood function.
        c (float): Second parameter of the log-likelihood function.
        y (np.ndarray): All observations.

    Returns:
        partial_c (float):
            The value of the partial derivative with respect to c of the log-likelihood function
            for the given parameters and observations.
    '''
    term_one_numerator = (k + 1)*(y**c)*np.array([cmath.log(y_val).real for y_val in y])
    term_one_denominator = y**(c) + 1
    term_one = np.divide(term_one_numerator, term_one_denominator)

    partial_c = np.sum(-term_one + 1 / c + np.array([cmath.log(y_val).real for y_val in y]))

    return partial_c

def log_likelihood_gradient(
    k : float,
    c : float,
    y : np.ndarray
) -> np.ndarray[float, float]:
    '''
    Computes the gradient for the log-likelihood function for a set of observations and a given set of parameters.

    Parameters:
        k (float): First parameter of the log-likelihood function.
        c (float): Second parameter of the log-likelihood function.
        y (np.ndarray): All observations.

    Returns:
        log_likelihood_gradient (np.ndarray([float, float])):
            The value of the gradient of the log-likelihood function for the given parameters and observations.
    '''
    log_likelihood_gradient = np.array([_k_partial(k, c, y), _c_partial(k, c, y)])
    return log_likelihood_gradient

def log_likelihood_function(
    k : float,
    c : float,
    y : np.ndarray
) -> float:
    '''
    Computes the log-likelihood function for a set of observations and a given set of parameters.

    Parameters:
        k (float): First parameter of the log-likelihood function.
        c (float): Second parameter of the log-likelihood function.
        y (np.ndarray): All observations.

    Returns:
        log_likelihood (float):
            The value of the log-likelihood function for the given parameters and observations.
    '''
    log_likelihood = np.sum(
        cmath.log(k).real +\
        cmath.log(c).real +\
        (c - 1)*np.array([cmath.log(y_val).real for y_val in y]) -\
        (k + 1)*np.array([cmath.log(1 + y_val**c).real for y_val in y])
    )

    return log_likelihood

def exact_line_search(
    _y : np.ndarray,
    y : np.ndarray,
    gradient : np.ndarray
) -> np.ndarray[float, float]:
    '''
    '''
    log_values = np.logspace(-2, 2, num = 4, base = 10.0)
    incremental_values = np.arange(0.2, 1.01, 0.2)

    step_sizes = np.outer(log_values, incremental_values).flatten()
    X, Y = np.meshgrid(step_sizes, step_sizes)
    meshed_step_sizes = np.stack((X, Y), axis = -1)
    meshed_step_sizes = meshed_step_sizes.reshape(len(step_sizes)**2, 2)

    exact_lines = np.array([
        [
            _y[0] + meshed_step_size[0] * gradient[0],
            _y[1] + meshed_step_size[1] * gradient[1]
        ]
        for meshed_step_size in meshed_step_sizes
    ])
    meshed_step_sizes = meshed_step_sizes[(exact_lines >= 0).all(axis = 1)]
    exact_lines = exact_lines[(exact_lines >= 0).all(axis = 1)]

    log_likelihoods = np.array([
        log_likelihood_function(
            exact_line[0], 
            exact_line[1],
            y
        ) for exact_line in exact_lines
    ])

    best_step = meshed_step_sizes[np.argmax(log_likelihoods)]
    return best_step

def accelerated_gradient_descent_using_exact_line_search(
    y : np.ndarray,
    tolerance : float = 1e-2
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    '''
    '''
    _x = np.array([0.5, 0.5])
    _x0 = _x.copy()
    _x00 = _x.copy()
    _y = np.array([0.5, 0.5])

    k = 1

    k_history = []
    k_history.append(_y[0])
    c_history = []
    c_history.append(_y[1])
    log_likelihood_history = []

    log_likelihood = log_likelihood_function(_y[0], _y[1], y)
    log_likelihood_history.append(log_likelihood)

    gradient = log_likelihood_gradient(_y[0], _y[1], y)

    while (np.dot(gradient.T, gradient) > tolerance):
        gradient = log_likelihood_gradient(_y[0], _y[1], y)
        meshed_step_size = exact_line_search(_y, y, gradient)
        
        _x = _y + meshed_step_size * gradient
        _y = _x0 + (k - 1) / (k + 2) * (_x0 - _x00)

        k_history.append(_x[0])
        c_history.append(_x[1])

        log_likelihood_old = log_likelihood
        log_likelihood = log_likelihood_function(_x[0], _x[1], y)

        log_likelihood_history.append(log_likelihood)

        _x0 = _x.copy()
        _x00 = _x0.copy()
        k += 1

    best_k = _x[0]
    best_c = _x[1]
    return (best_k, best_c, k_history, c_history, log_likelihood_history)