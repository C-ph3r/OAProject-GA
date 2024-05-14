
#Optuna

# Lists to plot the model comparison
fitness_scores = []

# Defining the objective function 
def objective(trial):
    population_size = trial.suggest_categorical('population_size', [25, 50, 100])
    num_gen = trial.suggest_categorical('num_gen', [50, 100, 200])
    mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.1, log=True)
    crossover_rate = trial.suggest_float('crossover_rate', 0.7, 0.9)
    selector= trial.suggest_categorical('selector', [selector_1, selector_2, selector_3])
    mutator= trial.suggest_categorical('mutator', [mutator_1, mutator_2, mutator_3])
    crossover= trial.suggest_categorical('crossover', [crossover_1, crossover_2, crossover_3])
   
    # Running genetic algorithm with the different parameters
    solution = GA(initializer,evaluator, selector, mutator, population_size, num_gen, crossover_rate, mutation_rate)
    
    # Evaluating the given solution
    distance = evaluate_solution(solution)
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

