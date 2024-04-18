import random
import numpy as np


def tournament_selection_min(ts):

    def inner_tournament(population, fitnesses):

        # randomly selecting ts number of individuals from the population:
        # or, more specifically, choosing the individuals from the pop. via their index
        pool = random.choices([i for i in range(len(population))], k=ts)

        # getting the fitnesses of the individuals of the given index
        pool_fits = [fitnesses[i] for i in pool]

        # finding out where in the pool fits the best fitness is
        best = np.argmin(pool_fits)

        # return the individual from the population whose index is the same as the index
        # in pool of the individual who was best in pool_fits
        return population[pool[best]]

    return inner_tournament


def tournament_selection_max(ts):

    def inner_tournament(population, fitnesses):
        # randomly selecting ts number of individuals from the population:
        # or, more specifically, choosing the individuals from the pop. via their index
        pool = random.choices([i for i in range(len(population))], k=ts)

        # getting the fitnesses of the individuals of the given index
        pool_fits = [fitnesses[i] for i in pool]

        # finding out where in the pool fits the best fitness is
        best = np.argmax(pool_fits)

        # return the individual from the population whose index is the same as the index
        # in pool of the individual who was best in pool_fits
        return population[pool[best]]

    return inner_tournament
