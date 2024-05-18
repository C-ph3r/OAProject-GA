import sys
sys.path.insert(0, '..')
from optuna.visualization import plot_optimization_history

from base.population import create_population, evaluate_population
from base.geo_gain_matrix import generate_matrix
from operators.selection_algorithms import SUS_selection, boltzmann_selection, tournament_selection
from operators.crossovers import order_xover, position_xover, cycle_xover, pmx_crossover
from operators.mutators import inversion_mutation, rgibnnm, swap_mutation
from algorithm.algorithm import GA
from utils.utils import get_n_elites
import optuna


# Stationary parameters
areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
initializer = create_population(areas=areas)
selection_pressure = 5



# Creating matrix list to have ground for comparision
matrixes = [generate_matrix(0.8, areas) for i in range(15)]

# Lists to plot the model comparison
fitness_scores = []

# Defining the objective function 
def objective(trial):
    pop_size = trial.suggest_categorical('pop_size', [25, 50, 100])
    n_gens = trial.suggest_categorical('n_gens', [10,20,50, 100, 150])
    mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.1, log=True)
    crossover_rate = trial.suggest_float('crossover_rate', 0.6, 0.9)
    selector= trial.suggest_categorical('selector', [SUS_selection,  tournament_selection])
    mutator= trial.suggest_categorical('mutator', [swap_mutation, inversion_mutation])
    crossover= trial.suggest_categorical('crossover', [ position_xover])
    n_elites = trial.suggest_int('n',1,3)

    elite_func = get_n_elites(n_elites)

    temp_list = []
    for matrix in matrixes:
        evaluator = evaluate_population(matrix)
    # Running genetic algorithm with the different parameters
        pop ,fit= GA(initializer, evaluator, 
                    selector, crossover, mutator, 
                    pop_size, n_gens, crossover_rate, mutation_rate,
                    elite_func, verbose=False, log_path=None, elitism=True, seed=42,
                    geo_matrix = matrix)
        
        temp_list.append(max(fit))
    # Returning the given solution
    
    return sum(temp_list)/len(temp_list)

def optimize_optuna(n_trials):
   print("hi")
# Running and tunning parameters with Optuna optimization
   study = optuna.create_study(direction='maximize')
   study.optimize(lambda trial: objective(trial), n_trials=n_trials)

   # Get the best parameters and their corresponding fitness
   best_params = study.best_params
   best_value = study.best_value

  
   # Plot the evolution of fitness values
   print("Best Parameters:", best_params)
   print("Best Distance:", best_value)
   fig = plot_optimization_history(study)
   fig.show()
   

optimize_optuna(15)