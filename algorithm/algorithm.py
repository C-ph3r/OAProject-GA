import random
from copy import deepcopy

from base.individual import *
from base.population import evaluate_population_ks_max

def GA(initializer, evaluator, selector, crossover, mutator,
       pop_size, n_gens, p_xo, p_m):


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
                o1, o2 = crossover(p1, p2) # TODO: WE STILL NEED TO IMPLEMENT XOVER.
            else:
                # reproduction
                o1, o2 = deepcopy(p1), deepcopy(p2)

            # mutating the offspring
            o1, o2 = mutator(o1, p_m), mutator(o2, p_m)

            # adding the offpring into the offspring population
            offspring.extend([o1, o2])

        # replacing the current population with the offpsring population
        population = offspring

        # evaluating the current population:
        pop_fit = evaluator(population)

    return population, pop_fit









