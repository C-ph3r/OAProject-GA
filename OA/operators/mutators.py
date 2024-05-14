import random
from base.individual_validator import individual_validator

# Function to perform inversion mutation
def inversion_mutation(route):
    '''
    Function that performs inversion mutation on a given route.

    In inversion mutation, a subsequence of cities within the route is randomly
    selected and its order is reversed, producing a mutated route.

    Inputs:
    - route (list): A list representing the route, where each element represents
      a city.

    Outputs: mutated_route (list): A mutated version of the input after inversion mutation
    '''
    validity = False

    while not validity:
        start_index = random.randint(0, len(route) - 2)
        end_index = random.randint(start_index + 1, len(route) - 1)
        mutated_route = route[:start_index] + route[start_index:end_index+1][::-1] + route[end_index+1:]
        validity = individual_validator(mutated_route)

    return mutated_route