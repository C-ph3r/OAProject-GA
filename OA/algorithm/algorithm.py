import sys
sys.path.insert(0, '..')
from operators.crossovers import  scx_xover
from operators.mutators import rgibnnm
from operators.selection_algorithms import boltzmann_selection
import csv
import numpy as np
from copy import deepcopy
import pandas as pd
import random
import matplotlib.pyplot as plt


def GA(initializer, evaluator, selector, crossover, mutator,
       pop_size, n_gens, p_xo, p_m, elite_func, geo_matrix,
       verbose=False, log_path=None, elitism=False, plot=False, seed=0):
    '''
    Run the Genetic Algorithm to solve a maximisation problem

    Inputs: initializer - Function to initialize the population
            evaluator - Function to evaluate the fitness of the population
            selector - Function for selection operation
            crossover - Function for crossover operation
            mutator - Function for mutation operation
            pop_size (int) - Size of the population
            n_gens (int) - Number of generations
            p_xo (float) - Probability of crossover
            p_m (float) - Probability of mutation
            elite_func - Function to determine elite individuals
            geo_matrix (DataFrame) - Geographic matrix for evaluations
            verbose (bool) - Whether to print verbose output
            log_path (str) - Path to log file
            elitism (bool) - Whether to use elitism
            seed (int) - Random seed for reproducibility
            plot (bool) - Whether to plot the results

    Outputs: pop, pop_fit - Final population and their fitness values.
    '''

    areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
    geo_matrix = pd.DataFrame(geo_matrix,index=areas, columns=areas)

    # 1. Setting up the seed:
    random.seed(seed)
    np.random.seed(seed)

    # Lists to store maximum and average fitness at each generation
    max_fitness_values = []
    avg_fitness_values = []

    # 2. Initializing the gen 0 population:
    population = initializer(pop_size)

    # 3. Evaluating the current population:
    pop_fit = evaluator(population)

    # 4. Main loop
    for gen in range(n_gens):

        # 4.1. Creating an empty offpsring population:
        offspring = []

        # 4.2. While the offspring population is not full:
        while len(offspring) < pop_size:
            # 4.2.1. Selecting the parents
            if selector == boltzmann_selection:
                temperature = max(0.1, 100 * (0.9 ** gen))
                p1 = selector(population, pop_fit, temperature)
                p2 = selector(population, pop_fit, temperature)
            else:
                p1 = selector(population, pop_fit)
                p2 = selector(population, pop_fit)

            # 4.2.2. Choosing between crossover and reproduction
            if random.random() < p_xo:
                if crossover == scx_xover:
                    o1, o2 = crossover(p1, p2, geo_matrix)
                else:
                    o1, o2 = crossover(p1, p2)
            else:   
                o1, o2 = deepcopy(p1), deepcopy(p2)

            # 4.2.3. Mutating the offspring
            # Dynamic mutation rate (decrease over generations):
            dyn_p_m = p_m * (1 - gen / n_gens)

            for o in [o1, o2]:
                if random.random() < dyn_p_m:
                    if mutator == rgibnnm:
                        o = mutator(o, geo_matrix)
                    else:
                        o = mutator(o)

                if o not in offspring and o not in population:
                    offspring.append(o)

        # 4.3. Making sure offspring population doesnt exceed pop_size:
        offspring = offspring[:pop_size]

        # 4.4. If elitism is used, apply it:
        if elitism:
            elite, best_fit = elite_func(population, pop_fit)
            if isinstance(elite[0], list):
                offspring.extend(elite[:pop_size - len(offspring)])  # Ensure size limit
            else:
                offspring.append(elite)

        # 4.5. Replacing the current population with the offpsring population
        population = offspring

        # 4.6. Evaluating the current population
        pop_fit = evaluator(population)

        # 4.7. Displaying and logging the generation results = the best fits
        new_elite, new_fit = elite_func(population, pop_fit)

        # 4.8. Calculating maximum and average fitness
        max_fitness = max(pop_fit)
        avg_fitness = np.mean(pop_fit)

        max_fitness_values.append(max_fitness)
        avg_fitness_values.append(avg_fitness)

        if verbose:
            print(f'     {gen}       |       {new_elite[0]} - {new_fit[0]}       ')
            print('-' * 32)

        if log_path is not None:
            with open(log_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([seed, gen, new_fit, new_elite])

    # 5. Plotting if enabled
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_gens), max_fitness_values, label='Maximum Fitness')
        plt.plot(range(n_gens), avg_fitness_values, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution of Fitness')
        plt.legend()
        plt.grid(True)
        plt.show()


    return population, pop_fit
