import numpy as np

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