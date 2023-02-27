# Standard Libraries
import os
import numpy as np
import cmath

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

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
    assert k > 0
    assert c > 0

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
    assert k > 0
    assert c > 0

    term_one_numerator = (k + 1)*(y**c)*np.array([cmath.log(y_val).real for y_val in y])
    term_one_denominator = y**(c + 1)
    term_one = np.divide(term_one_numerator, term_one_denominator)

    partial_c = np.sum(-term_one + 1 / c + np.array([cmath.log(y_val).real for y_val in y]))

    return partial_c

def log_likelihood_gradient(
    k : float,
    c : float,
    y : np.ndarray
) -> list[float, float]:
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
    assert k > 0
    assert c > 0

    log_likelihood_gradient = [_k_partial(k, c, y), _c_partial(k, c, y)]
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
    assert k > 0
    assert c > 0

    log_likelihood = np.sum(
        cmath.log(k).real +\
        cmath.log(c).real +\
        (c - 1)*np.array([cmath.log(y_val).real for y_val in y]) -\
        (k + 1)*np.array([cmath.log(1 + y_val**c).real for y_val in y])
    )

    return log_likelihood

def exact_line_search(
    k : float,
    c : float,
    y : np.ndarray,
    gradient : np.ndarray
) -> float:
    '''
    '''
    log_values = np.logspace(-10, 1, num = 10, base = 10.0)
    incremental_values = np.arange(0.1, 1.1, 0.1)

    step_sizes = np.outer(log_values, incremental_values).flatten()

    exact_lines = np.array([
        [
            k + step_size * gradient[0],
            c + step_size * gradient[1]
        ]
        for step_size in step_sizes
    ])
    step_sizes = step_sizes[(exact_lines >= 0).all(axis = 1)]
    exact_lines = exact_lines[(exact_lines >= 0).all(axis = 1)]

    log_likelihoods = np.array([
        log_likelihood_function(
            exact_line[0], 
            exact_line[1],
            y
        ) for exact_line in exact_lines
    ])

    best_step = step_sizes[np.argmax(log_likelihoods)]
    return best_step

def gradient_descent_using_exact_line_search(
    y : np.ndarray,
    tolerance : float = 1e-8
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    '''
    '''
    _k = 0.5
    _c = 0.5
    
    k_history = []
    k_history.append(_k)
    c_history = []
    c_history.append(_c)
    log_likelihood_history = []

    log_likelihood = log_likelihood_function(_k, _c, y)
    log_likelihood_history.append(log_likelihood)

    gradient = np.array(log_likelihood_gradient(_k, _c, y))

    max_iteration = 30
    iteration = 1
    while (np.dot(gradient.T, gradient) > tolerance) and (iteration < max_iteration):
        gradient = np.array(log_likelihood_gradient(_k, _c, y))
        # print(gradient)
        # step_size = exact_line_search(_k, _c, y, gradient)
        step_size = 1e-5
        
        _k = _k + step_size * gradient[0]
        k_history.append(_k)

        _c = _c + step_size * gradient[1]
        c_history.append(_c)

        log_likelihood_old = log_likelihood
        log_likelihood = log_likelihood_function(_k, _c, y)

        # print(log_likelihood)

        log_likelihood_history.append(log_likelihood)
        iteration += 1

    best_k = _k
    best_c = _c
    return (best_k, best_c, k_history, c_history, log_likelihood_history)

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question2.csv'))

    y = np.genfromtxt(file_directory, delimiter = ',').flatten()

    max_iterations = 10
    best_k, best_c, k_history, c_history, log_likelihood_history = gradient_descent_using_exact_line_search(y, max_iterations)

    # Plot Log-Likelihood
    iterations = np.arange(0, len(log_likelihood_history))
    sns.lineplot(x = iterations, y = log_likelihood_history)

    plt.title('Log-Likelihood Progression Using Gradient Ascent with Exact Line Search')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'gradient_ascent.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

    # Plot K and C History
    iterations = np.arange(0, len(k_history))
    sns.lineplot(x = iterations, y = k_history)
    sns.lineplot(x = iterations, y = c_history)

    plt.title('Values of K and C Using Gradient Ascent with Exact Line Search')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Values')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'k_and_c_history_gradient_ascent.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

    print(best_k, best_c)