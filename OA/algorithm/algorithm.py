import csv
import numpy as np
from copy import deepcopy
from base.individual import *


def GA(initializer, evaluator, selector, crossover, mutator,
       pop_size, n_gens, p_xo, p_m, elite_func, verbose=False,
       log_path=None, elitism=True, seed=0):

    # setting up the seed:
    random.seed(seed)
    np.random.seed(seed)

    if elite_func is None:
        raise Exception("Without a proper elite function I cannot work. Humph! *grumpy sounds*")

    # TODO: do i need to think about seeds?

    # initializing the gen 0 population:
    population = initializer(pop_size)

    # evaluating the current population:
    pop_fit = evaluator(population)

    for gen in range(n_gens):

        # creating an empty offpsring population:
        offspring = []

        # while the offspring population is not full:
        while len(offspring) < pop_size:
            # selecting the parents
            p1, p2 = selector(population, pop_fit), selector(population, pop_fit)

            # choosing between xover and reproduction
            if random.random() < p_xo:
                # xover
                o1, o2 = crossover(p1, p2)
            else:
                # reproduction
                o1, o2 = deepcopy(p1), deepcopy(p2)

            # mutating the offspring
            o1, o2 = mutator(o1, p_m), mutator(o2, p_m)

            # adding the offpring into the offspring population
            offspring.extend([o1, o2])

        # making sure offspring population doesnt exceed pop_size
        while len(offspring) > pop_size:
            offspring.pop()


        # if elitism, make sure the elite of the population is inserted into the next generation
        if elitism:
            elite, best_fit = elite_func(population, pop_fit)
            offspring[-1] = elite # adding the elite, unchanged into the offspring population

        # replacing the current population with the offpsring population
        population = offspring

        # evaluating the current population:
        pop_fit = evaluator(population)

        # displaying and logging the generation results
        new_elite, new_fit = elite_func(population, pop_fit)

        if verbose:
            print(f'     {gen}       |       {new_fit}       ')
            print('-' * 32)

        if log_path is not None:
            with open(log_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([seed, gen, new_fit, new_elite])

    return population, pop_fit