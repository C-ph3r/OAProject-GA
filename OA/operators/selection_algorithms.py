import random
import numpy as np
import math

def tournament_selection(pop:list, pop_fit:list, ts=1):
    '''
    Tournament selection algorithm
    
    Inputs: pop (list) - List of individuals in the population
            pop_fit (list) - List of fitness values of the population given
        
    Outputs: list - selected individual
    '''
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


def SUS_selection(pop:list, pop_fit:list, n_sel = 1):
    '''
    Stochastic Universal Sampling (SUS) selection algorithm
    
    Inputs: pop (list) - List of individuals in the population
            pop_fit (list) - List of fitness values of the population given
        
    Outputs: list - selected individual
    '''
    fitness_sum = sum(pop_fit)

    # creating selection pointers
    pointer_distance = fitness_sum / n_sel # Distance between pointers
    start = random.uniform(0, pointer_distance) # Random start
    pointers = [start + i * pointer_distance for i in range(n_sel)]    # Evenly spaced
    
    current_fitness = 0
    index = 0
    
    for pointer in sorted(pointers):
        # Sum fitness values until the pointer threshold
        while current_fitness < pointer:
            current_fitness += pop_fit[index]
            index += 1
        # Add the corresponding individual to the list 
        # When only 1 is to be selected, return only the individual
        selected = pop[index - 1]
    
    return selected


def boltzmann_selection(pop:list, pop_fit:list, temperature = 0.5):
    '''
    Boltzmann Selection algorithm for selecting individuals based on entropy
    
    Inputs: pop (list) - List of individuals in a population
            pop_fit (list) - List of fitness values of the population given
            temperature (float) - Temperature parameter (controls the amount of randomness)
        
    Output: list - Selected individual
    '''
    pop_size = len(pop)

    # Calculate the probability of selection for each individual
    max_fit = max(pop_fit)
    adjusted_fitness = [(fit - max_fit) / temperature for fit in pop_fit]

    # Calculate Boltzmann probabilities for each individual
    exp_fitness = [np.exp(fit) for fit in adjusted_fitness]
    fitness_sum = sum(exp_fitness)
    probabilities = [exp_fit / fitness_sum for exp_fit in exp_fitness]

    # Perform selection based on Boltzmann probabilities
    selected = random.choices(pop, weights=probabilities)[0]
    
    return selected
    