# Standard Libraries
import os
import numpy as np
import cmath
import time

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Imports
from gradient_ascent import gradient_descent_using_exact_line_search
from accelerated_gradient_ascent import accelerated_gradient_descent_using_exact_line_search
from stochastic_gradient_ascent import stochastic_gradient_descent_using_exact_line_search
from newtons_method import newtons_method_using_exact_line_search

def optimizer_1_report(
    y : np.ndarray
) -> None:
    '''
    '''
    tolerance = 1e-1
    best_k, best_c, k_history, c_history, log_likelihood_history = gradient_descent_using_exact_line_search(y, tolerance)

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

def optimizer_2_report(
    y : np.ndarray
) -> None:
    '''
    '''
    tolerance = 1e-1
    best_k, best_c, k_history, c_history, log_likelihood_history = accelerated_gradient_descent_using_exact_line_search(y, tolerance)

    # Plot Log-Likelihood
    iterations = np.arange(0, len(log_likelihood_history))
    sns.lineplot(x = iterations, y = log_likelihood_history)

    plt.title('Log-Likelihood Progression Using Accelerated \nGradient Ascent with Exact Line Search')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'accelerated_gradient_ascent.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

    # Plot K and C History
    iterations = np.arange(0, len(k_history))
    sns.lineplot(x = iterations, y = k_history)
    sns.lineplot(x = iterations, y = c_history)

    plt.title('Values of K and C Using Accelerated \nGradient Ascent with Exact Line Search')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Values')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'k_and_c_history_accelerated_gradient_ascent.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

def optimizer_3_report(
    y : np.ndarray
) -> None:
    '''
    '''
    tolerance = 1e-1
    best_k, best_c, k_history, c_history, log_likelihood_history = stochastic_gradient_descent_using_exact_line_search(y, tolerance)

    print(best_k)
    print(best_c)

    # Plot Log-Likelihood
    iterations = np.arange(0, len(log_likelihood_history))
    sns.lineplot(x = iterations, y = log_likelihood_history)

    plt.title('Log-Likelihood Progression Using Stochastic \nGradient Ascent with Exact Line Search')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'stochastic_gradient_ascent.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

    # Plot K and C History
    iterations = np.arange(0, len(k_history))
    sns.lineplot(x = iterations, y = k_history)
    sns.lineplot(x = iterations, y = c_history)

    plt.title('Values of K and C Using Stochastic \nGradient Ascent with Exact Line Search')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Values')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'k_and_c_history_stochastic_gradient_ascent.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

def optimizer_4_report(
    y : np.ndarray
) -> None:
    '''
    '''
    best_k, best_c, k_history, c_history, log_likelihood_history = newtons_method_using_exact_line_search(y)

    # Plot Log-Likelihood
    iterations = np.arange(0, len(log_likelihood_history))
    sns.lineplot(x = iterations, y = log_likelihood_history)

    plt.title('Log-Likelihood Progression Using \nNewtons Method with Exact Line Search')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'newtons_method.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

    # Plot K and C History
    iterations = np.arange(0, len(k_history))
    sns.lineplot(x = iterations, y = k_history)
    sns.lineplot(x = iterations, y = c_history)

    plt.title('Values of K and C Using Stochastic \nNewtons Method with Exact Line Search')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Values')

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'k_and_c_history_newtons_method.png'))
    plt.savefig(file_directory, dpi = 100)

    plt.clf()
    plt.cla()

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question2.csv'))

    y = np.genfromtxt(file_directory, delimiter = ',').flatten()

    # start_time = time.time()
    # optimizer_1_report(y)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print(fr'Total time taken for optimizer 1: {total_time}')

    # start_time = time.time()
    # optimizer_2_report(y)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print(fr'Total time taken for optimizer 2: {total_time}')

    # start_time = time.time()
    # optimizer_3_report(y)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print(fr'Total time taken for optimizer 3: {total_time}')

    start_time = time.time()
    optimizer_4_report(y)
    end_time = time.time()
    total_time = end_time - start_time
    print(fr'Total time taken for optimizer 4: {total_time}')