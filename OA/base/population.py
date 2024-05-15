from base.individual import generate_route
from base.fitness_function import individual_fitness

def create_population(individual_size):

    def generate_pop(pop_size):

        return [generate_route(individual_size)() for _ in range(pop_size)]

    return generate_pop

def evaluate_population(geo_matrix):

    def pop_evaluation(population):

        return [individual_fitness(route, geo_matrix) for route in population]

    return pop_evaluation

