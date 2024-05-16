import sys
sys.path.insert(0, '..')

from algorithm.algorithm import GA
from base.population import create_population, evaluate_population
from operators.selection_algorithms import SUS_selection, tournament_selection, boltzmann_selection
from operators.crossovers import order_xover
from operators.mutators import inversion_mutation, rgibnnm
from utils.utils import get_n_elites
from base.geo_gain_matrix import generate_matrix

from tqdm import trange


# Stationary parameters
areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
geo_gain_matrix = generate_matrix(0.8, areas)
initializer = create_population(areas_list=areas)
evaluator = evaluate_population(geo_matrix=geo_gain_matrix)
elite_func = get_n_elites(3)
selection_pressure = 5
xover = order_xover
mutator = inversion_mutation
selector =  tournament_selection
log_path = 'log\testing_logging.csv'

# evolution based parameters
pop_size = 20
n_gens = 100
p_xo = 0.8
p_m = 0.3
n_elites = 2


for seed in trange(10):

   GA(initializer=initializer,
      evaluator=evaluator,
      selector=selector,
      crossover=xover,
      mutator=mutator,
      pop_size=pop_size,
      n_gens=n_gens,
      p_xo=p_xo,
      p_m=p_m,
      geo_matrix=geo_gain_matrix,
      verbose=True,
      log_path=None, elitism=True,
      elite_func=get_n_elites(n_elites), seed=seed)