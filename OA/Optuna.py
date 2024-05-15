from base.population import create_population, evaluate_population
from base.fitness_function import individual_fitness
from base.geo_gain_matrix import generate_matrix
from operators.selection_algorithms import SUS_selection, boltzmann_selection, tournament_selection
from operators.crossovers import order_xover, position_xover
from operators.mutators import inversion_mutation, rgibnnm, swap_mutation
from algorithm.algorithm import GA
from utils.utils import get_n_elites
import optuna
import matplotlib.pyplot as plt


# Stationary parameters
areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
geo_gain_matrix = generate_matrix(0.8, areas)
initializer = create_population(areas_list=areas)
evaluator = evaluate_population(geo_gain_matrix)
elite_func = get_n_elites(3)
selection_pressure = 5

# Lists to plot the model comparison
fitness_scores = []

# Defining the objective function 
def objective(trial):
    pop_size = trial.suggest_categorical('pop_size', [25, 50, 100])
    n_gens = trial.suggest_categorical('n_gens', [50, 100, 200])
    mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.1, log=True)
    crossover_rate = trial.suggest_float('crossover_rate', 0.7, 0.9)
    selector= trial.suggest_categorical('selector', [SUS_selection, boltzmann_selection(0.5), tournament_selection])
    mutator= trial.suggest_categorical('mutator', [swap_mutation, inversion_mutation, rgibnnm])
    crossover= trial.suggest_categorical('crossover', [order_xover, position_xover])
   
    # Running genetic algorithm with the different parameters
    solution = GA(initializer, evaluator, 
                  selector, crossover, mutator, 
                  pop_size, n_gens, crossover_rate, mutation_rate,
                  elite_func, verbose=False, log_path=False, elitism=True, seed=42,
                  geo_matrix = geo_gain_matrix)
    
    # Evaluating the given solution
    distance = individual_fitness(solution)
    fitness_scores.append(distance)
    
    return distance

def optimize_optuna(n_trials):
# Running and tunning parameters with Optuna optimization
   study = optuna.create_study(direction='minimize')
   study.optimize(objective, n_trials=n_trials)

   # Get the best parameters and their corresponding fitness
   best_params = study.best_params
   best_value = study.best_value

   print("Best Parameters:", best_params)
   print("Best Distance:", best_value)
   # Plot the evolution of fitness values
   plt.plot(fitness_scores, label='Fitness Scores')
   plt.xlabel('Trials')
   plt.ylabel('Fitness Score')
   plt.legend()
   plt.show()

optimize_optuna(1)