import numpy as np

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
    partial_k = np.sum(1 / k - np.log(y^c + 1))
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
    numerator = 1 + y^c + c * np.log(y) + c*k*y^c*np.log(y)
    denominator = c * (1 + y^c)

    partial_c = np.divide(numerator, denominator)
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
    numerator = k * c * y^(c - 1)
    denominator = (1 + y)^(k + 1)
    fraction = np.log(np.divide(numerator, denominator))

    log_likelihood = np.sum(fraction)

    return log_likelihood

def exact_line_search(
    k : float,
    c : float,
    y : np.ndarray,
    gradient : np.ndarray,
    learning_rate : float
) -> float:
    '''
    '''
    step_sizes = np.linspace(0.001, 1.0, num = 1000)

    log_likelihoods = np.array([
        log_likelihood_function(
            k - learning_rate * gradient[0], 
            c - learning_rate * gradient[1],
            y
        ) for learning_rate in step_sizes
    ])

    best_step = step_sizes[np.argmax(log_likelihoods)]
    return best_step

def gradient_descent_using_exact_line_search(
    y : np.ndarray,
    learning_rate : float,
    max_iterations : int
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    '''
    '''
    _k = 0
    _c = 0
    
    k_history = []
    c_history = []
    log_likelihood_history = []

    log_likelihood = log_likelihood_function(_k, _c, y)
    log_likelihood_history.append(log_likelihood)

    for i in range(max_iterations):
        gradient = log_likelihood_gradient(_k, _c, y)
        step_size = exact_line_search(_k, _c, y, gradient, learning_rate)

        _k = _k - step_size * gradient[0]
        k_history.append(_k)

        _c = _c - step_size * gradient[1]
        c_history.append(_c)

        log_likelihood = log_likelihood_function(_k, _c, y)
        log_likelihood_history.append(log_likelihood)

    return (_k, _c, k_history, c_history, log_likelihood_history)

if __name__ == '__main__':
    y = None
    learning_rate = None
    max_iterations = None
    _k, _c, k_history, c_history, log_likelihood_history = gradient_descent_using_exact_line_search(y, learning_rate, max_iterations)
