import random

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

def generate_route(areas):
    '''
    Function that generates 1 random route by shuffing the list of areas to visit
    Input:  areas (list of strings) - Possible areas to visit
    Output: individual (list of strings) - order in which the player visits the areas
    '''
    return ['D'] + random.choices(areas, len(areas), replace=False)