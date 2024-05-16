import sys
sys.path.insert(0, '..')

from base.population import create_population, evaluate_population
from operators.selection_algorithms import tournament_selection
areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
initializer = create_population(areas_list=areas)
evaluator = evaluate_population()

# 2. Initializing the gen 0 population:
population = initializer(4)

# 3. Evaluating the current population:
pop_fit = evaluate_population(population)

print(pop_fit)

'''# 4.1. Creating an empty offpsring population:
offspring = []

# 4.2. While the offspring population is not full:
while len(offspring) < 4:
    # 4.2.1. Selecting the parents
    p1 = tournament_selection(population, pop_fit)
    p2 = tournament_selection(population, pop_fit)

print(offspring)'''