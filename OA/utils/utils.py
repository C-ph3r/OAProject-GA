import numpy as np

def get_n_elites_max(n):

    def get_elite(population, pop_fit):

        # getting the best n elites:

        bests_i = np.argsort(pop_fit)[-n:]

        # getting the fitnesses of the best n elites:

        return [population[i] for i in bests_i], [pop_fit[i] for i in bests_i]  # returning the list of elites and their list of fitnesses

    return get_elite

