import sys
sys.path.insert(0, '..')

from exp_functions import *

base_mutator = rgibnnm
base_crossover = scx_xover
base_selector = boltzmann_selection

'''crossovers = [order_xover, position_xover, scx_xover]
crossover_results = compare_crossovers(base_mutator, crossovers, base_selector)
plot_results(crossover_results, 'Comparison of Crossovers')'''

'''selectors = [boltzmann_selection, SUS_selection, tournament_selection]
selector_results = compare_selectors(base_mutator, base_crossover, selectors)
plot_results(selector_results, 'Comparison of Selectors')'''

mutators = [rgibnnm, swap_mutation, inversion_mutation]
mutator_results = compare_mutators(mutators, base_crossover, base_selector)
plot_results(mutator_results, 'Comparison of Mutators')