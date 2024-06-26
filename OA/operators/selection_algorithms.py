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


import numpy as np

def boltzmann_selection(pop, pop_fit, temperature=0.5):
    '''
    Boltzmann Selection algorithm for selecting individuals based on entropy
    
    Inputs: pop (list) - List of individuals in a population
            pop_fit (list) - List of fitness values of the population given
            temperature (float) - Temperature parameter (controls the amount of randomness)
        
    Output: list - Selected individual
    '''
    pop_fit = np.array(pop_fit)

    # Normalize fitness values to avoid overflow
    max_fit = np.max(pop_fit)
    if max_fit > 0:
        norm_fit = pop_fit / max_fit
    else:
        norm_fit = pop_fit
    
    # Exponential fitness values scaled by temperature
    scaled_fit = np.exp(norm_fit / max(temperature, 1e-10))  # Avoid division by zero or very small temperature
    
    # Boltzmann probabilities
    probabilities = scaled_fit / np.sum(scaled_fit)
    
    # Ensure probabilities do not contain NaN values
    if np.any(np.isnan(probabilities)):
        probabilities = np.ones_like(probabilities) / len(probabilities)
    
    selected_index = np.random.choice(len(pop), p=probabilities)
    
    return pop[selected_index]

    