import random
import numpy as np
import math

def SUS_selection():
    """
    Stochastic Universal Sampling (SUS) selection algorithm
    
    Inputs: pop (list) - List of individuals in the population
            pop_fit (list) - List of fitness values of the population given
        
    Outputs: list - selected individual
    """

    def SUS(pop, pop_fit):
        fitness_sum = sum(pop_fit)
        num_selections = 1

        # creating selection pointers
        pointer_distance = fitness_sum / num_selections # Distance between pointers
        start = random.uniform(0, pointer_distance) # Random start
        pointers = [start + i * pointer_distance for i in range(num_selections)]    # Evenly spaced
        
        selected = []
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
    
    return SUS


def boltzmann_selection(temperature):
    """
    Boltzmann Selection algorithm for selecting individuals based on entropy
    
    Inputs: pop (list) - List of individuals in a population
            pop_fit (list) - List of fitness values of the population given
            temperature (float) - Temperature parameter (controls the amount of randomness)
        
    Output: list - Selected individuals
    """

    def boltzmann(pop, pop_fit):
        num_selections = 1
        # Calculate Boltzmann probabilities for each individual
        fitness_sum = sum(math.exp(fit / temperature) for fit in pop_fit)
        probabilities = [math.exp(individual.fitness / temperature) / fitness_sum for individual in pop]
        
        # Perform selection based on Boltzmann probabilities
        selected = []
        for _ in range(num_selections):
            selected = random.choices(pop, weights=probabilities)[0]
        # in this case only 1 is selected
        
        return selected
    
    return boltzmann