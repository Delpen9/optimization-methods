# Standard Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _partial_beta1(
    d1 : float,
    y1 : float,
    di : float,
    yi : float,
    beta1 : float,
    beta2 : float
) -> float:
    '''
    '''
    numerator = y1**2*(np.e**(beta2*di) - 1)
    denominator = (beta1*np.e**(beta2*di) - y1*np.e**(beta2*di) + y1)**2

    partial_beta1 = numerator / denominator
    return partial_beta1

def _partial_beta2(
    d1 : float,
    y1 : float,
    di : float,
    yi : float,
    beta1 : float,
    beta2 : float
) -> float:
    '''
    '''
    numerator = di*beta1*y1*(beta1 - y1)*np.e**(beta2*di)
    denominator = ((beta1 - y1)*np.e**(beta2*di) + y1)**2

    partial_beta2 = numerator / denominator
    return partial_beta2

def jacobian(
    d1 : float,
    y1 : float,
    d : np.ndarray,
    y : np.ndarray,
    beta1 : float,
    beta2 : float
) -> np.ndarray:
    '''
    '''
    jacobian = []
    for di, yi in zip(d, y):
        partial_beta1 = _partial_beta1(d1, y1, di, yi, beta1, beta2)
        partial_beta2 = _partial_beta2(d1, y1, di, yi, beta1, beta2)
        jacobian.append([partial_beta1, partial_beta2])
    jacobian = np.array(jacobian)
    return jacobian

def decomposed_loss_function(
    d1 : float,
    y1 : float,
    d : np.ndarray,
    y : np.ndarray,
    beta1 : float,
    beta2 : float
) -> np.ndarray:
    _g = y - (y1*beta1)/(y1 + (beta1 - y1)*np.e**(beta2*d))
    return _g.T

def loss_function(
    d1 : float,
    y1 : float,
    d : np.ndarray,
    y : np.ndarray,
    beta1 : float,
    beta2 : float
) -> float:
    '''
    '''
    numerator = y1*beta1
    denominator = y1 + (beta1 - y1)*np.e**(-beta2*d)
    loss_function = np.sum((y - numerator / denominator)**2)
    return loss_function

def gauss_newton_adaptive_step_size(
    d : np.ndarray,
    y : np.ndarray,
    beta1 : float,
    beta2 : float,
    tolerance : float = 1e-2,
    damping_factor : float = 1e-10
) -> tuple[float, float]:
    '''
    '''
    alpha = 1

    _jacobian = jacobian(d1, y1, d, y, beta1, beta2)
    _g = decomposed_loss_function(d1, y1, d, y, beta1, beta2)

    omega = -np.linalg.solve(
        np.linalg.inv(_jacobian.T @ _jacobian + damping_factor * np.eye(2)),
        _jacobian.T @ _g
    )

    while (np.amax(np.abs(omega)) > tolerance):
        _jacobian = jacobian(d1, y1, d, y, beta1, beta2)
        _g = decomposed_loss_function(d1, y1, d, y, beta1, beta2)

        omega = -np.linalg.solve(
            np.linalg.inv(_jacobian.T @ _jacobian + damping_factor * np.eye(2)),
            _jacobian.T @ _g
        )

        print(omega)

        _loss = loss_function(d1, y1, d, y, beta1, beta2)
        new_loss = _loss + 1 # Only for starting the while loop

        while new_loss > _loss:
            beta1_new = beta1 - alpha * omega[0]
            beta2_new = beta2 - alpha * omega[1]

            new_loss = loss_function(d1, y1, d, y, beta1_new, beta2_new)

            alpha *= 0.1

        beta1 = beta1_new
        beta2 = beta2_new

        alpha = alpha**0.5

    best_beta1 = beta1
    best_beta2 = beta2
    return (best_beta1, best_beta2)

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question3-1.csv'))

    data = np.genfromtxt(file_directory, delimiter = ',')
    d = data[:, 0].copy()
    y = data[:, 1].copy()

    d1 = d[0]
    y1 = y[0]

    beta1 = 0.5
    beta2 = 0.5

    best_beta1, best_beta2 = gauss_newton_adaptive_step_size(
        d,
        y,
        beta1,
        beta2,
    )

    print(best_beta1, best_beta2)