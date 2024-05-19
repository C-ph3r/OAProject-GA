import sys
sys.path.insert(0, '..')
from operators.crossovers import  scx_xover, pmx_xover
from operators.mutators import rgibnnm
import csv
import numpy as np
from copy import deepcopy
import pandas as pd
import random


def GA(initializer, evaluator, selector, crossover, mutator,
       pop_size, n_gens, p_xo, p_m, elite_func, geo_matrix,
       verbose=False, log_path=None, elitism=False, seed=0):

    areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
    geo_matrix = pd.DataFrame(geo_matrix,index=areas, columns=areas)

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
                if crossover == scx_xover:
                    o1, o2 = crossover(p1, p2, geo_matrix)
                else:
                    o1, o2 = crossover(p1, p2)
            else:
                o1, o2 = deepcopy(p1), deepcopy(p2)

            # 4.2.3. Mutating the offspring
            for o in [o1, o2]:
                if random.random() < p_m:
                    if mutator == rgibnnm:
                        o = mutator(o, geo_matrix)
                    else:
                        o = mutator(o)

            # 4.2.4. Adding the offpring into the offspring population
            offspring.extend([o1, o2])

        # 4.3. Making sure offspring population doesnt exceed pop_size
        while len(offspring) > pop_size:
            offspring.pop()

        # 4.4. If elitism is used, apply it
        if elitism:
            elite, best_fit = elite_func(population, pop_fit)
            if isinstance(elite[0], list):
                offspring.extend(elite)  # If elite is a list of lists, extend offspring
            else:
                offspring.append(elite)


        #4.4.1 Assuring that the offspring list has same size as parent
        while len(offspring) > pop_size:
            offspring.pop(0)

        

        # 4.5. Replacing the current population with the offpsring population
        population = offspring

        # 4.6. Evaluating the current population
        pop_fit = evaluator(population)

        # 4.7. Displaying and logging the generation results = the best fits
        new_elite, new_fit = elite_func(population, pop_fit)

        if verbose:
            print(f'     {gen}       |       {new_elite[0]} - {new_fit[0]}       ')
            print('-' * 32)

        if log_path is not None:
            with open(log_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([seed, gen, new_fit, new_elite])


    return population, pop_fit
