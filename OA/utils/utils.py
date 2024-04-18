import numpy as np

def get_elite_min(population, pop_fit):
    # returns the elite and the elite fitness
    return population[np.argmin(pop_fit)], min(pop_fit)

def get_elite_max(population, pop_fit):
    # returns the elite and the elite fitness

    return population[np.argmax(pop_fit)], max(pop_fit)
