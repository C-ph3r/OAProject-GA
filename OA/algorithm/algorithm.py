import sys
sys.path.insert(0, '..')

import csv
import numpy as np
from copy import deepcopy
import pandas as pd
import random


def GA(initializer, evaluator, selector, crossover, mutator,
       pop_size, n_gens, p_xo, p_m, elite_func, geo_matrix,
       verbose=False, log_path=None, elitism=False, seed=0):

    geo_matrix = pd.DataFrame(geo_matrix)

    # 1. Setting up the seed:
    random.seed(seed)
    np.random.seed(seed)

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
            p1 = selector(population, pop_fit)
            p2 = selector(population, pop_fit)

            # 4.2.2. Choosing between crossover and reproduction
            if random.random() < p_xo:
                o1, o2 = crossover(p1, p2)
            else:
                o1, o2 = deepcopy(p1), deepcopy(p2)

            # 4.2.3. Mutating the offspring
            if random.random() < p_m:
                o1 = mutator(o1)
            elif random.random() < p_m:
                o2 = mutator(o2)

            # 4.2.4. Adding the offpring into the offspring population
            offspring.extend([o1, o2])

        # 4.3. Making sure offspring population doesnt exceed pop_size
        while len(offspring) > pop_size:
            offspring.pop()

        # 4.4. If elitism is used, apply it
        if elitism:
            elite, best_fit = elite_func(population, pop_fit)
            offspring[-1] = elite # adding the elite, unchanged into the offspring population

        # 4.5. Replacing the current population with the offpsring population
        population = offspring

        # 4.6. Evaluating the current population
        pop_fit = evaluator(population)

        # 4.7. Displaying and logging the generation results = the best fits
        new_elite, new_fit = elite_func(population, pop_fit)

        if verbose:
            print(f'     {gen}       |       {new_elite} - {new_fit}       ')
            print('-' * 32)

        if log_path is not None:
            with open(log_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([seed, gen, new_fit, new_elite])


    return population, pop_fit