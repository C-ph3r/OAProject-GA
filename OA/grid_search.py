from base.population import create_population, evaluate_population

# Stable parameters (cannot be optimised)
# List of areas the player can visit
areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
initializer = create_population(len(areas))
evaluator = evaluate_population

# Parameters to optimise
selector = tournament_selection_max(10)  # high selection pressure with a larger tournament size :)

# function based parameters
xover = [xover1, xover2, xover3]
mutator = [mut1, mut2, mt3]

# evolution based parameters
pop_size = 20
n_gens = 100
p_xo = 0.8
p_m = 0.3
n_elites = 2

# Function
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