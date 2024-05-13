import random
import numpy as np

def tournament_selection(ts):

    def inner_tournament(pop, pop_fit):
        # randomly selecting ts number of individuals from the population:
        # or, more specifically, choosing the individuals from the population via their index
        pool = random.choices([i for i in range(len(pop))], k=ts)

        # getting the pop_fit of the individuals of the given index
        pool_fits = [pop_fit[i] for i in pool]

        # finding out where in the pool fits the best fitness is
        best = np.argmax(pool_fits)

        # return the individual from the population whose index is the same as the index
        # in pool of the individual who was best in pool_fits
        return pop[pool[best]]

    return inner_tournament
