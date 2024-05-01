import numpy as np


def get_elite_min(population, pop_fit):
    # returns the elite and the elite fitness
    return population[np.argmin(pop_fit)], min(pop_fit)


def get_elite_max(population, pop_fit):
    # returns the elite and the elite fitness

    return population[np.argmax(pop_fit)], max(pop_fit)


def get_n_elites_max(n):

    def get_elite(population, pop_fit):

        # getting the best n elites:

        bests_i = np.argsort(pop_fit)[-n:]

        # getting the fitnesses of the best n elites:

        return [population[i] for i in bests_i], [pop_fit[i] for i in bests_i]  # returning the list of elites and their list of fitnesses

    return get_elite


f = get_n_elites_max(2)

print(f([[1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1]], [10, 20, 60, 30]))
