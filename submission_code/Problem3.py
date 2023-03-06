# Standard Libraries
import os
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

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
    numerator = y1**2*np.e**(beta2*di)*(np.e**(beta2*di) - 1)
    denominator = (beta1 + y1*(np.e**(beta2*di) - 1))**2

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
    denominator = (beta1 + y1*(np.e**(beta2*di) - 1))**2

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
    _g = y - (y1*beta1)/(y1 + (beta1 - y1)*np.e**(-beta2*d))
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

def loss_function_left_term(
    y : np.ndarray
) -> float:
    '''
    '''
    loss_function_left_term = np.sum(y)
    return loss_function_left_term
    
def loss_function_right_term(
    d1 : float,
    y1 : float,
    d : np.ndarray,
    beta1 : float,
    beta2 : float
) -> float:
    '''
    '''
    numerator = y1*beta1
    denominator = y1 + (beta1 - y1)*np.e**(-beta2*d)
    loss_function_right_term = np.sum(numerator / denominator)
    return loss_function_right_term

def get_scatter(
    d1 : float,
    y1 : float,
    d : np.ndarray,
    y : np.ndarray,
    beta1 : float,
    beta2 : float,
) -> np.ndarray:
    def right_term(
        d1 : float,
        y1 : float,
        di : float,
        beta1 : float,
        beta2 : float
    ) -> float:
        numerator = y1*beta1
        denominator = y1 + (beta1 - y1)*np.e**(-beta2*di)
        loss_function_right_term = numerator / denominator
        return loss_function_right_term

    all_right_terms = np.array([right_term(d1, y1, d, beta1, beta2) for di in d]).T
    all_y_values = np.expand_dims(np.arange(0, 101), axis = 0).T

    print(all_right_terms)

    scatter_values = np.hstack((all_y_values, all_right_terms))

    return scatter_values

def gauss_newton_adaptive_step_size(
    d : np.ndarray,
    y : np.ndarray,
    beta1 : float,
    beta2 : float,
    tolerance : float = 1e-2,
    damping_factor : float = 1e-10,
    max_iterations : float = 100
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    '''
    alpha = 1e+2

    beta_1_history = []
    beta_1_history.append(beta1)
    beta_2_history = []
    beta_2_history.append(beta2)

    left_term_history = []
    left_term_history.append(loss_function_left_term(y))
    right_term_history = []
    right_term_history.append(loss_function_right_term(d1, y1, d, beta1, beta2))

    loss_function_history = []
    loss_function_history.append(loss_function(d1, y1, d, y, beta1, beta2))

    _jacobian = jacobian(d1, y1, d, y, beta1, beta2)
    _g = decomposed_loss_function(d1, y1, d, y, beta1, beta2)

    omega = -np.linalg.solve(
        np.linalg.inv(_jacobian.T @ _jacobian + damping_factor * np.eye(2)),
        _jacobian.T @ _g
    )

    iteration = 1
    while (np.amax(np.abs(omega)) > tolerance) and (iteration < max_iterations):
        _jacobian = jacobian(d1, y1, d, y, beta1, beta2)
        _g = decomposed_loss_function(d1, y1, d, y, beta1, beta2)

        omega = -np.linalg.solve(
            np.linalg.inv(_jacobian.T @ _jacobian + damping_factor * np.eye(2)),
            _jacobian.T @ _g
        )

        _loss = loss_function(d1, y1, d, y, beta1, beta2)

        left_term_history.append(loss_function_left_term(y))
        right_term_history.append(loss_function_right_term(d1, y1, d, beta1, beta2))

        loss_function_history.append(loss_function(d1, y1, d, y, beta1, beta2))

        new_loss = _loss + 1

        while new_loss > _loss:
            beta1_new = beta1 - alpha * omega[0]
            beta2_new = beta2 - alpha * omega[1]

            new_loss = loss_function(d1, y1, d, y, beta1_new, beta2_new)

            alpha *= 0.1

        beta1 = beta1_new
        beta2 = beta2_new

        beta_1_history.append(beta1)
        beta_2_history.append(beta2)

        alpha = alpha**0.5

        iteration += 1

    best_beta1 = beta1
    best_beta2 = beta2

    beta_1_history = np.array(beta_1_history)
    beta_2_history = np.array(beta_2_history)

    left_term_history = np.array(left_term_history)
    right_term_history = np.array(right_term_history)

    loss_function_history = np.array(loss_function_history)

    return (best_beta1, best_beta2, beta_1_history, beta_2_history, left_term_history, right_term_history, loss_function_history)

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

    best_beta1, best_beta2, beta_1_history, beta_2_history, left_term_history, right_term_history, loss_function_history = gauss_newton_adaptive_step_size(
        d,
        y,
        beta1,
        beta2,
    )

    # Beta 1 and Beta 2
    iterations = np.arange(0, len(beta_1_history))
    sns.lineplot(x = iterations, y = beta_1_history)
    sns.lineplot(x = iterations, y = beta_2_history)

    plt.title('Values of Beta 1 and Beta 2 \nUsing the Gauss-Newton Method')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Values')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'beta_1_2_history_gauss_newton_method.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

    # y
    iterations = np.arange(0, len(left_term_history))
    sns.lineplot(x = iterations, y = left_term_history)

    plt.title('Values of y \nUsing the Gauss-Newton Method')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Values')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'y_history_gauss_newton_method.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

    # right term
    iterations = np.arange(0, len(right_term_history))
    sns.lineplot(x = iterations, y = right_term_history)

    plt.title('Values of Right Term \nUsing the Gauss-Newton Method')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Values')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'right_term_history_gauss_newton_method.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

    # loss function
    iterations = np.arange(0, len(loss_function_history))
    ax = sns.lineplot(x = iterations, y = loss_function_history)
    ax.set(yscale = 'log')

    plt.title('Values of Loss Function \nUsing the Gauss-Newton Method')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Values')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'loss_function_history_gauss_newton_method.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

    # scatter plot
    scatter_data = get_scatter(d1, y1, d, y, best_beta1, best_beta2)

    plt.scatter(scatter_data[:, 0], scatter_data[:, 1])

    plt.xlabel('Y values')
    plt.ylabel('Right Term Values')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'y_vs_right_term_gauss_newton_method.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()