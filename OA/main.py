from algorithm.algorithm import GA
from base.population import create_population, evaluate_population
from operators.selection_algorithms import tournament_selection_max
from operators.crossovers import one_point_xover
from operators.mutators import single_bit_mutation
from utils.utils import get_n_elites

from tqdm import trange

# List of areas the player can visit
areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']

# function based parameters
initializer = create_population(len(areas))

evaluator = evaluate_population()

selector = tournament_selection_max(10)  # high selection pressure with a larger tournament size :)

xover = one_point_xover

mutator = single_bit_mutation

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

