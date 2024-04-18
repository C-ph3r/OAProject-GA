import random

def generate_solution_ks(size):

    def generate_sol():
        return [random.randint(0, 1) for _ in range(size)]

    return generate_sol


def get_fitness_ks_max(values, volumes, capacity):

    def get_fitness(sol):

        # getting the total value of the items in the solution
        fitness = sum([values[i] * sol[i] for i in range(len(sol))])

        # getting the total volume these items occupy
        total_vol = sum([volumes[i] * sol[i] for i in range(len(sol))])

        return fitness if total_vol <= capacity else 0

    return get_fitness


def get_fitness_ks_min(values, volumes, capacity): # TODO: finish/change this
    
    def get_fitness(sol):
        # getting the total value of the items in the solution
        fitness = sum([values[i] * sol[i] for i in range(len(sol))])

        # getting the total volume these items occupy
        total_vol = sum([volumes[i] * sol[i] for i in range(len(sol))])

        return fitness # we will talk about this soon ;)

    return get_fitness