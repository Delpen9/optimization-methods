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

def log_likelihood_hessian(

) -> np.ndarray:
    return None

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