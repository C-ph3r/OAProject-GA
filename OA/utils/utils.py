import numpy as np


def get_elite_min(population, pop_fit):
    # returns the elite and the elite fitness
    return population[np.argmin(pop_fit)], min(pop_fit)


def get_elite_max(population, pop_fit):
    # returns the elite and the elite fitness

    return population[np.argmax(pop_fit)], max(pop_fit)

def check_next_item(lst, target, next_item):
    '''
    Function used in the fitness calculation to find if the next string in a list
    corresponds to a predetermined one
    Inputs: lst (list) - list in which to search
            target - item we are looking to find
            next_item - item we want to check if is right after 'target'
    Output: boolean value of the result
    '''
    target_index = lst.index(target)
    if target_index < len(lst) - 1 and lst[target_index + 1] == next_item:
        # If the next item is the one we want, return True
        return True
    else:
        return False

def get_n_elites_max(n):

  def get_elite(population, pop_fit):

    # getting the best n elites:

    bests_i = np.argsort(pop_fit)[-n:]

    # getting the fitnesses of the best n elites:

    return [population[i] for i in bests_i], [pop_fit[i] for i in bests_i]  # returning the list of elites and their list of fitnesses

  return get_elite


f = get_n_elites_max(2)

print(f([[1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1]], [10, 20, 60, 30]))
