import sys
sys.path.insert(0, '..')

from base.population import create_population, evaluate_population
from operators.selection_algorithms import tournament_selection
from base.geo_gain_matrix import generate_matrix

areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
matrix = generate_matrix(0.8, areas)
initializer = create_population(areas_list=areas)
evaluator = evaluate_population(matrix)

# 2. Initializing the gen 0 population:
population = initializer(4)

# 3. Evaluating the current population:
pop_fit = evaluator(population)

# 4.2. While the offspring population is not full:
p1 = tournament_selection(population, pop_fit)
p2 = tournament_selection(population, pop_fit)

print(p1, p2)