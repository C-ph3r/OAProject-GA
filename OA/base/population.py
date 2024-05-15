from base.individual import generate_route
from base.fitness_function import individual_fitness

def create_population(areas_list):

    def generate_pop(pop_size):

        return [generate_route(areas_list)() for _ in range(pop_size)]

    return generate_pop

def evaluate_population(geo_matrix):

    def pop_evaluation(population):

        return [individual_fitness(route, geo_matrix) for route in population]

    return pop_evaluation

