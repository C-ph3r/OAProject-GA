import sys
sys.path.insert(0, '..')
from base.population import create_population, evaluate_population
from operators.selection_algorithms import SUS_selection, boltzmann_selection, tournament_selection
from operators.crossovers import  position_xover, scx_xover, pmx_xover, order_xover
from operators.mutators import inversion_mutation, rgibnnm, swap_mutation
from algorithm.algorithm import GA
from utils.utils import get_n_elites_max
import optuna
import pandas as pd


# Stationary parameters
areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
initializer = create_population(areas=areas)
selection_pressure = 5



# Importing matrixes previously created, to have grounds for comparision
matrixes_file = pd.ExcelFile("matrixes.xlsx")

matrixes = []
for i in range(15):
    temp = pd.read_excel(matrixes_file,f"{i+1}", index_col=0)
    matrixes.append(temp)



# Defining the objective function 
def objective(trial):
    pop_size = trial.suggest_categorical('pop_size', [25,  100,150])
    n_gens = trial.suggest_categorical('n_gens', [10, 150])
    mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.1, log=True)
    crossover_rate = trial.suggest_float('crossover_rate', 0.6, 0.9)
    selector= trial.suggest_categorical('selector', [ SUS_selection, boltzmann_selection, tournament_selection])
    mutator= trial.suggest_categorical('mutator', [swap_mutation, inversion_mutation, rgibnnm])
    crossover= trial.suggest_categorical('crossover', [ order_xover])
    n_elites = trial.suggest_int('n',1,10)

    elite_func = get_n_elites_max(n_elites)

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
# Running and tunning parameters with Optuna optimization
   study = optuna.create_study(direction='maximize')
   study.optimize(lambda trial: objective(trial), n_trials=n_trials)

   # Get the best parameters and their corresponding fitness
   best_params = study.best_params
   best_value = study.best_value

  

   # Plot the evolution of fitness values
   print("Best Parameters:", best_params)
   print("Best Distance:", best_value)
   fig1 = optuna.visualization.plot_optimization_history(study)
   fig1.show()

   fig2 = (optuna.visualization.plot_param_importances(study))
   fig2.show()



optimize_optuna(15)