import numpy as np

def get_elite_min(population, pop_fit):
    # returns the elite and the elite fitness
    return population[np.argmin(pop_fit)], min(pop_fit)

def get_elite_max(population, pop_fit):
    # returns the elite and the elite fitness

    return population[np.argmax(pop_fit)], max(pop_fit)

def get_n_elites_max(n:int):

    def get_elite(population, pop_fit):
        # best n elites
        return [#list of elites], [#list of their fits]

    return get_elite
