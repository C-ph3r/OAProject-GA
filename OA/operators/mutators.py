import random
from base.individual_validator import individual_validator
import numpy as np
from copy import deepcopy


def swap_mutation(tour):
    """
    Apply swap mutation to a tour in the TSP population.
    
    Parameters:
    - tour: List representing the tour (list of city indices)
    
    Returns:
    - mutated_tour: List representing the mutated tour
    """
    # Make a copy of the tour to avoid modifying the original
    mutated_tour = tour.deepcopy()
    
    # Select two distinct random indices in the tour
    idx1, idx2 = random.sample(range(len(mutated_tour)), 2)
    
    # Swap the cities at the selected indices
    mutated_tour[idx1], mutated_tour[idx2] = mutated_tour[idx2], mutated_tour[idx1]
    
    # Ensure that all cities are visited exactly once
    mutated_set = set(mutated_tour)
    for city in tour:
        if city not in mutated_set:
            mutated_tour.append(city)
            break
    
    return mutated_tour
=======

def inversion_mutation(route):
    '''
    Function that performs inversion mutation on a given route.
    In inversion mutation, a sequence of cities within the route is randomly
    selected and its order is reversed.

    Inputs: route (list) - Order in which the player visits the areas
    Outputs: mutated_route (list) - A mutated version of the input after inversion mutation
    '''
    validity = False

    while not validity:
        start_index = random.randint(0, len(route) - 2)
        end_index = random.randint(start_index + 1, len(route) - 1)
        mutated_route = route[:start_index] + route[start_index:end_index+1][::-1] + route[end_index+1:]
        validity = individual_validator(mutated_route)

    return mutated_route

def rgibnnm(route, geo_gain_matrix):
    '''
    Function that performs RGIBNNM mutation on a given route.

    Inputs: route (list) - Order in which the player visits the areas
            geo_gain_matrix (list of lists) - Geo gain per pair of areas
    Outputs: mutated_route (list) - A mutated version of the input after inversion mutation
    '''
    validity = False

    while not validity:
        # Selecting a random area
        indA = random.randint(0, len(route) - 1)
        areaA = route[indA]

        # Finding the one with the maximum geo gain from it
        areaB = max((c for c in route if c != areaA), key=lambda c: geo_gain_matrix[areaA][c])

        # From the areas with most geo gain from it
        range_of_cities = [c for c in route if abs(c - areaB) >= np.mean(geo_gain_matrix[areaA]) and c != areaA]

        if range_of_cities:
            # Select one area to swap with areaA
            areaC = random.choice(range_of_cities)
            mutated = deepcopy(route)
            mutated[route.index(areaA)] = areaC
            mutated[route.index(areaC)] = areaA
        
        validity = individual_validator(mutated)

    return mutated

