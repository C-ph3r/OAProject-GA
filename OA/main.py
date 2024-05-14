from algorithm.algorithm import GA
from base.population import create_population, evaluate_population
from operators.selection_algorithms import SUS_selection, tournament_selection, boltzmann_selection
from operators.crossovers import order_xover
from operators.mutators import inversion_mutation, rgibnnm
from utils.utils import get_n_elites
from base.geo_gain_matrix import generate_matrix

from tqdm import trange

# List of areas the player can visit
areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']

# function based parameters
geo_gain_matrix = generate_matrix(0.8, areas)

initializer = create_population(len(areas))

evaluator = evaluate_population(geo_gain_matrix)

selector = tournament_selection(10)  # high selection pressure with a larger tournament size :)

xover = order_xover()

mutator = inversion_mutation(rgibnnm(geo_gain_matrix=geo_gain_matrix)) 
   # nota: este mutator tem de ter este parâmetro pre-set

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
      verbose=True,
      log_path="log/test_log.csv", elitism=True,
      elite_func=get_n_elites(n_elites), seed=seed)

