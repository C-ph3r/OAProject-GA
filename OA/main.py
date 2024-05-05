from algorithm.algorithm import GA
from base.population import create_population_ks, evaluate_population_ks_max
from operators.selection_algorithms import tournament_selection_max
from operators.crossovers import one_point_xover
from operators.mutators import single_bit_mutation
from utils.utils import get_elite_max
from base.ks_data import *

from tqdm import trange

# function based parameters
initializer = create_population_ks(individual_size=len(values))

evaluator = evaluate_population_ks_max(values=values, volumes=volumes, capacity=capacity)

selector = tournament_selection_max(10)  # high selection pressure with a larger tournament size :)

xover = one_point_xover

mutator = single_bit_mutation

# evolution based parameters
pop_size = 20
n_gens = 100
p_xo = 0.8
p_m = 0.3


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
      elite_func=get_elite_max, seed=seed)

