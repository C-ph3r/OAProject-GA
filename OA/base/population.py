from base.individual import *
from fitness_function import *

def create_population(individual_size):

    def generate_pop(pop_size):

        return [generate_route(individual_size)() for _ in range(pop_size)]

    return generate_pop

def evaluate_population(geo_matrix):

    def pop_evaluation(population):

        return [individual_fitness(route, geo_matrix) for route in population]

    return pop_evaluation

