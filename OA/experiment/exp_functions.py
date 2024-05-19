from algorithm.algorithm import GA
from base.population import create_population, evaluate_population
from utils.utils import get_n_elites_max
from base.geo_gain_matrix import generate_matrix
import matplotlib.pyplot as plt

def run_experiment(crossover, mutator, selector):
    '''
    Run the Genetic Algorithm with standard parameters except for operators
    Inputs: crossover, mutator, selector (functions) - Genetic operators to test
    Outputs: pop_fit (list) - List of fitnesses to plot
    '''
    areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
    geo_matrix = generate_matrix(areas)

    # Run the GA with specified parameters
    population, pop_fit = GA(
        initializer=create_population(areas),
        evaluator=evaluate_population(geo_matrix),
        selector=selector,
        crossover=crossover,
        mutator=mutator,
        pop_size=50,
        n_gens=100,
        p_xo=0.7,
        p_m=0.1,
        elite_func=get_n_elites_max(2),
        geo_matrix=geo_matrix,
        verbose=False,
        log_path=None,
        elitism=True,
        seed=0,
        plot=False
    )
    return pop_fit  # Return fitness values for plotting

def compare_mutators(mutators, crossover, selector):
    '''
    Compare different mutator methods while keeping crossover and selector constant.

    Inputs: mutators (list) - List of mutator functions to compare
            constant_crossover - Crossover function to keep constant
            constant_selector - Selector function to keep constant

    Output: Dictionary with mutator names as keys and fitness values as values
    '''

    results = {}
    for mutator in mutators:
        key = mutator.__name__
        print(f'Running GA with mutator: {key}')
        pop_fit = run_experiment(mutator, crossover, selector)
        results[key] = pop_fit
    return results

def compare_selectors(mutator, crossover, selectors):
    '''
    Compare different selector methods while keeping crossover and mutator constant.

    Inputs: mutator - Mutator function to keep constant
            crossover - Crossover function to keep constant
            selectors (list) - List of selector functions to compare

    Output: Dictionary with mutator names as keys and fitness values as values
    '''

    results = {}
    for selector in selectors:
        key = selector.__name__
        print(f'Running GA with selector: {key}')
        pop_fit = run_experiment(mutator, crossover, selector)
        results[key] = pop_fit
    return results

def compare_crossovers(mutator, crossovers, selector):
    '''
    Compare different selector methods while keeping crossover and mutator constant.

    Inputs: mutator - Mutator function to keep constant
            crossovers (list) - List of crossover functions to compare
            selector - Selector function to keep constant

    Output: Dictionary with mutator names as keys and fitness values as values
    '''

    results = {}
    for crossover in crossovers:
        key = selector.__name__
        print(f'Running GA with crossover: {key}')
        pop_fit = run_experiment(mutator, crossover, selector)
        results[key] = pop_fit
    return results

def plot_results(results, title):
    """
    Plot the results of the fitness values for different methods.

    Inputs: results (dict) - Dictionary with method names as keys and fitness values as values.
            title (str) - Title for the plot.
    """
    plt.figure(figsize=(12, 8))
    for label, fitness in results.items():
        plt.plot(fitness, label=label)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title(title)
    plt.legend()
    plt.show()