import sys
sys.path.insert(0, '..')

from base.individual import generate_possible_route
from base.fitness_function import individual_fitness

def create_population(areas:list) -> list:
    '''
    Creates a valid population
    Input: areas (list) - List of possible areas to visit
    Output: list - List of individuals (routes) of size pop_size
    '''
    def generate_pop(pop_size):

        return [generate_possible_route(areas) for _ in range(pop_size)]

    return generate_pop

def evaluate_population(geo_matrix):
    '''
    Returns a list of fitnesses
    '''
    
    def pop_evaluation(population):

        return [individual_fitness(route, geo_matrix) for route in population]

    return pop_evaluation

