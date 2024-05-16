import numpy as np

def check_next_item(lst, target, next_item):
    '''
    Function used in the viability calculation to find if the next string in a list
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

def get_n_elites(n):
  '''
  Elitism function. Selects the best fitting individuals
  Inputs: n - number of elites 
        pop - population 
        pop_fit - the fit values of the population
  Outputs: list of elites, list of their fitnesses
  '''

  def get_elite(pop, pop_fit):

    # getting the best n elites:
    bests_i = np.argsort(pop_fit)[-n:]

    # getting the list of elites and their list of fitnesses
    return [pop[i] for i in bests_i], [pop_fit[i] for i in bests_i]

  return get_elite