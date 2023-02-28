# Standard Libraries
import os
import numpy as np
import cmath

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Imports
from gradient_ascent import gradient_descent_using_exact_line_search

def optimizer_1_report(
    y : np.ndarray
) -> None:
    '''
    '''
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

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question2.csv'))

    y = np.genfromtxt(file_directory, delimiter = ',').flatten()

    optimizer_1_report(y)