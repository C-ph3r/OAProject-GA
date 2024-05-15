import random
from individual_validator import individual_validator
from copy import deepcopy

'''
A solution for this problem should be a vector of size 11, the number of
areas the player passes through in any given session.

Each index in the vector corresponds to the codename of the area the player visits
in order. If an area is after the final 'D', it is considered to not be visited.

Example solution: ['D', 'FC', 'G', 'QS', 'DV', 'QG', 'CS', 'SN', 'RG', 'D', 'KS']

Areas:
Dirtmouth (D)
Forgotten Crossroads (FC)
Greenpath (G)
Queen's Station (QS)
Queen's Gardens (QG)
City Storerooms (CS)
King's Station (KS)
Resting Grounds (RG)
Distant Village (DV)
Stag Nest (SN)
'''


def generate_route(areas: list) -> list:
    '''
    Function that generates 1 random route by randomly selecting and removing items from the list of areas to visit
    Input:  areas (list of strings) - Possible areas to visit
    Output: individual (list of strings) - order in which the player visits the areas
    '''
    areas_copy = areas[1:]  # Not using 'D'
    random_route = ['D']  # Initialize the random route list with 'D' at the beginning
    
    while areas_copy:  # Until all areas are visited
        # Selecing an area, adding to the route and removing from the copy
        random_area = random.sample(areas_copy, 1)[0]
        random_route.append(random_area)
        areas_copy.remove(random_area)
    
    random_route.append('D')  # Add 'D' at the end of the random route
    return random_route


print(generate_route(['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']))

def generate_possible_route(areas:list) -> list:
    '''
    Function that generates 1 possible route by shuffing the list of areas to visit, then
    verifying if it is actually possible given the constraints
    Input:  areas (list of strings) - Possible areas to visit
    Output: individual (list of strings) - order in which the player visits the areas
    '''
    valitidy = False
    while not valitidy:
        route = generate_route(areas)
        valitidy = individual_validator(route)
    
    return route