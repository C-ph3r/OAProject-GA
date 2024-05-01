import numpy as np

def get_elite_min(population, pop_fit):
    # returns the elite and the elite fitness
    return population[np.argmin(pop_fit)], min(pop_fit)

def get_elite_max(population, pop_fit):
    # returns the elite and the elite fitness

    return population[np.argmax(pop_fit)], max(pop_fit)

def get_n_elites_max(n: int):
    
    def get_elite(population, pop_fit):
        elites = []
        fits = []

        for i in range(n):
            curr_elite, curr_fit = get_elite_max(population, pop_fit)
            elites.append(curr_elite)
            fits.append(curr_fit)
            elite_index = np.argmax(pop_fit)
            population.pop(elite_index)
            pop_fit.pop(elite_index)
            
        return elites, fits

    return get_elite
