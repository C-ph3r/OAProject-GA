from base.individual import *

def create_population_ks(individual_size):

    def generate_pop(pop_size):

        return [generate_solution_ks(individual_size)() for _ in range(pop_size)]

    return generate_pop

def evaluate_population_ks_max(values, volumes, capacity):

    def pop_evaluation(population):

        # getting the individual evaluating function in accordance to the problem definition
        evaluate_ind = get_fitness_ks_max(values, volumes, capacity)

        # evaluating each solution in the population
        return [evaluate_ind(sol) for sol in population]

    return pop_evaluation

def evaluate_population_ks_min(values, volumes, capacity):

    def pop_evaluation(population):
        # getting the individual evaluating function in accordance to the problem definition
        evaluate_ind = get_fitness_ks_min(values, volumes, capacity)

        return [evaluate_ind(sol) for sol in population]

    return pop_evaluation

