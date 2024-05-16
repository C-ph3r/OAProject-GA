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
      # Get the indices that would sort the population by fitness in descending order
      best_indices = np.argsort(pop_fit)[-n:][::-1]
      
      # Get the list of elites and their fitnesses
      elites = [pop[i] for i in best_indices]
      elite_fitnesses = [pop_fit[i] for i in best_indices]
      
      return elites, elite_fitnesses

  return get_elite