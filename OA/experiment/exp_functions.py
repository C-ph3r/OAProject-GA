import sys
sys.path.insert(0, '..')

# from algorithm.algorithm import GA
import csv
import numpy as np
from copy import deepcopy
import pandas as pd
import random
import matplotlib.pyplot as plt
from pandas import ExcelWriter
import random

def individual_validator(route:list) -> bool:
    '''
    Function that returns True if the individual is valid, given the conditions
    Input:  route (list of strings) - Order in which areas are visited
    Output: bool - Whether or not the route is valid
    '''
    # Condition 1: All sessions must begin and end in Dirtmouth (D)
    if route[0] != 'D' or route[-1] != 'D':
        return False
    
    # Condition 2: City Storerooms (CS) cannot be visited right after Queen's Gardens (QG)
    if  route[route.index('QG')+1] == 'CS':
        return False
    
    # Condition 3: Resting Grounds (RG) can only be reached in the last half of the session.
    if 'RG' in route[:len(route)//2]:
        return False
    
    # If no conditions apply, then the route is valid
    return True

def can_it_skip_KS(route:list) -> bool:
    '''
    Function that returns True if this route meets the conditions to skip 'KS'
    Input: route (list of strings) - Order in which areas are visited
    Output: bool - Whether or not Ks can be skipped
    '''
    # If 'DV' is visited right after 'QS', the route may skil 'KS'
    index_qs = route.index('QS')
    if index_qs < len(route) - 1 and route[index_qs + 1] == 'DV':
        return True
    else:
        return False

def swap_mutation(route):
    """
    Apply swap mutation to a route in the TSP population.
    
    Parameters:
    - route: List representing the route (list of city indices)
    
    Returns:
    - mutated_route: List representing the mutated route
    """
    # Make a copy of the route to avoid modifying the original
    mutated_route = deepcopy(route)
    
    # Select two distinct random indices in the route
    idx1, idx2 = random.sample(range(len(mutated_route)), 2)
    
    # Swap the cities at the selected indices
    mutated_route[idx1], mutated_route[idx2] = mutated_route[idx2], mutated_route[idx1]
    
    # Ensure that all cities are visited exactly once
    mutated_set = set(mutated_route)
    for city in route:
        if city not in mutated_set:
            mutated_route.append(city)
            break
    
    return mutated_route

def inversion_mutation(route):
    '''
    Function that performs inversion mutation on a given route.
    In inversion mutation, a sequence of cities within the route is randomly
    selected and its order is reversed.

    Inputs: route (list) - Order in which the player visits the areas
    Outputs: mutated_route (list) - A mutated version of the input after inversion mutation
    '''
    validity = False

    while not validity:
        start_index = random.randint(0, len(route) - 2)
        end_index = random.randint(start_index + 1, len(route) - 1)
        mutated_route = route[:start_index] + route[start_index:end_index+1][::-1] + route[end_index+1:]
        validity = individual_validator(mutated_route)

    return mutated_route

def rgibnnm(route:list, geo_gain_matrix:pd.DataFrame):
    '''
    Function that performs RGIBNNM mutation on a given route.

    Inputs:
    - route (list): Order in which the player visits the areas.
    - geo_gain_matrix (DataFrame): Geo gain per pair of areas.
    
    Outputs:
    - mutated_route (list): A mutated version of the input after inversion mutation.
    '''
    validity = False

    while not validity:
        # Selecting a random area
        indA = random.randint(0, len(route) - 1)
        areaA = route[indA]

        # Finding the one with the maximum geo gain from it
        areaB = max((c for c in route if c != areaA), key=lambda c: geo_gain_matrix.loc[areaA, c])

        # From the areas with most geo gain from it
        range_of_cities = [c for c in route if geo_gain_matrix.loc[areaB, c] >= np.mean(geo_gain_matrix.loc[areaB, :]) and c != areaA]

        if range_of_cities:
            # Select one area to swap with areaA
            areaC = random.choice(range_of_cities)
            mutated = deepcopy(route)
            mutated[route.index(areaA)] = areaC
            mutated[route.index(areaC)] = areaA
        
            validity = individual_validator(mutated)

    return mutated

def order_xover(p1,p2):
    '''
    Performs order crossover on 2 parents. Steps:
    1- Select a randomly sized number of characters from the middle of both parents
    2- Assign the belt from parent 1 to child 1, and the belt from parent 2 to child 2
    3- Understand the characters that are already in each child, and those from the parents that are not yet in the child
    4- Assign the unused characters from parent 2 (in their original order), to child 1 and vice versa

    input:
    p1 (list): first parent on which to perform crossover
    p2 (list): second parent on which to perform crossover

    output:
    c1, c2 (lists): Crossed over children, with same lenght as the parents
    '''
    c1 = [-1 for i in p1]
    c2 = [-1 for i in p1]

    #Choosing size of the middle belt
    size_middle_belt = random.randint(1, len(p1)//2 -1)


    #Assigning the middle belt to children
    c1[size_middle_belt: -size_middle_belt] = p1[size_middle_belt: -size_middle_belt]
    c2[size_middle_belt: -size_middle_belt] = p2[size_middle_belt: -size_middle_belt]

    #Acessing the values present in the belts
    belt_1 = p1[size_middle_belt: -size_middle_belt]
    belt_2 = p2[size_middle_belt: -size_middle_belt]

    #Obtaining the values that are not in the belts, crossed over
    from_1 = [char for char in p1 if char not in belt_2]
    from_2 = [char for char in p2 if char not in belt_1]

    #Completing the lists with values from the other parent
    for i in range(len(c1)):
        if len(from_2) == 0:
            break
        if c1[i] == -1:
            c1[i] = from_2[0]
            from_2.pop(0)

    for i in range(len(c2)):
        if len(from_1) == 0:
            break
        if c2[i] == -1:
            c2[i] = from_1[0]
            from_1.pop(0)

    #Checking child validity, if any isn't valid, substitute it by the parent
    if not individual_validator(c1):
        c1 = p1
    if not individual_validator(c2):
        c2 = p2

    return c1,c2

def position_xover(p1,p2):
    '''
    Performs position crossover on two parent solutions
    Steps:
    1 - Chooses a random number of values to keep
    2 - Transfers said values in their original positions from p1 to c1 and p2 to c2
    3 - Transfers the remaining values crossed over from p2 to c1 and p1 to c2

    
    input:
    p1 (list): firstj parent on which to perform crossover
    p2 (list): second parent on which to perform crossover

    output:
    c1, c2 (lists): Crossed over children, with same lenght as the parents
    '''

    n_positions = random.randint(1, len(p1)-2)

    c1 = [-1 for i in p1]
    c2 = [-1 for i in p1]

    sample_1 = random.sample(p1,n_positions)
    sample_2 = random.sample(p2,n_positions)

    values_p1 = [val for val in p1 if val not in sample_2]
    values_p2 = [val for val in p2 if val not in sample_1]

    for i in range(len(p1)):
        if p1[i] in sample_1:
            c1[i] = p1[i]

        if p2[i] in sample_2:
            c2[i] = p2[i]


    for i in range(len(p1)):
        if c1[i] == -1:
            c1[i] = values_p2[0]
            values_p2.pop(0)

        if c2[i] == -1:
            c2[i] = values_p1[0]
            values_p1.pop(0)

    #Checking child validity, if any isn't valid, substitute it by the parent
    if not individual_validator(c1):
        c1 = p1
    if not individual_validator(c2):
        c2 = p2

    return c1,c2

def scx_xover(p1, p2, geo_matrix):
    '''
    Performs Sequential Constructive Crossover on two parent solutions. Steps:
    1 - Calls the generate_offspring function to generate the first offspring
    2 - Calls the same function again to generate the second offspring

    input:
    p1 (list): First parent on which to perform crossover.
    p2 (list): Second parent on which to perform crossover.
    geo_matrix (matrix): Matrix of the geo gain from one city to another.

    output:
    offspring1, offspring2 (lists): Crossed over children, with same length as the parents.
    '''

    def next_city(current_city, unvisited, p1, p2, geo_matrix):
        '''
        Identifies the best next city to be appended to the offspring. Steps:
        1 - Identifies the index of the current city on both of the parents
        2 - Searches through the parents for the next city not already visited
        3 - Compares the 2 options given by the 2 parents 

        input:
        current_city (string): Name of the current city.
        unvisited (list): List of cities left to be visited.
        p1 (list): First parent on which to perform crossover.
        p2 (list): Second parent on which to perform crossover.
        geo_matrix (matrix): Matrix of the geo gain from one city to another.

        output:
        next_city_p1/next_city_p2 (string): City to where it was consider best to go to next.
        '''

        # Get the indices of the current city in both parents
        idx1 = p1.index(current_city)
        idx2 = p2.index(current_city)

        # Making adjustments in the parents, so that a solution can be returned in both
        p1_search=p1[idx1:]+ p1[:idx1+1]
        p2_search= p2[idx2:]+ p2[:idx2+1]

        # Find the next unvisited city in parent 1 and parent 2
        next_city_p1 = None
        i= 0
        while next_city_p1 is None:
            if p1_search[i] in unvisited:
                next_city_p1 = p1_search[i]
            else:
                i+=1
        next_city_p2 = None
        j=0
        while next_city_p2 is None:
            if p2_search[j] in unvisited:
                next_city_p2 = p2_search[j]
            else:
                j+=1

        # Choose the one city that yields higher geo gain
        if geo_matrix.loc[current_city, next_city_p1] > geo_matrix.loc[current_city, next_city_p2]:
            return next_city_p1
        else:
            return next_city_p2


    def generate_offspring(start_parent, other_parent, geo_matrix):
        '''
        Generates a new offspring. Steps:
        1 - Transfers the first city of the parents directly to the offspring
        2 - Calls next_city function to define which is the best city to be transfered to the offspring
        Repeats step 2 until the offspring is the same lenght as the parents

        input:
        start_parent (list): First parent on which to perform crossover (Gives the first city).
        other_parent (list): Second parent on which to perform crossover.
        geo_matrix (matrix): Matrix of the geo gain from one city to another.

        output:
        offspring (list): Crossed over child, with same length as the parents.
        '''

        # Start from the first city of start_parent
        current_city = start_parent[0]
        offspring = [current_city]
        unvisited = set(start_parent) - {current_city}

        while unvisited:
            next_city_candidate = next_city(current_city, unvisited, start_parent, other_parent, geo_matrix)
            if next_city_candidate is None:
                print("error")
            offspring.append(next_city_candidate)
            unvisited.remove(next_city_candidate)
            current_city = next_city_candidate

        return offspring

    # Generating two offsprings
    offspring1 = generate_offspring(p1, p2, geo_matrix)
    offspring2 = generate_offspring(p2, p1, geo_matrix)

    if not individual_validator(offspring1):
        offspring1 = p1
    if not individual_validator(offspring2):
        offspring2 = p2

    return offspring1, offspring2

def pmx_xover(p1, p2):
    '''
    Performs partially mapped crossover on 2 parents. Steps:
    1: Select a random swath of consecutive alleles
    2: Copy the swath from Parent 1 to Child 1 and from Parent 2 to Child 2
    3: Fill the rest of each child according to the mapping created in fill_child
    4: Complete the children with the genes from the opposite parent

    input:
    p1 (list): first parent on which to perform crossover
    p2 (list): second parent on which to perform crossover

    output:
    c1, c2 (lists): Crossed over children, with the same length as the parents
    '''
    size = len(p1)
    c1 = [-1] * size
    c2 = [-1] * size
    
    # Choosing indexes for the middle belt
    start, end = sorted(random.sample(range(size), 2))
    
    # Assigning the belt to the children
    c1[start:end + 1] = p1[start:end + 1]
    c2[start:end + 1] = p2[start:end + 1]
    # Helper function to fill the child according to PMX mapping rules
    def fill_child(child, p1, p2, start, end):
        '''
        Fills children according to PMX mapping rules, iterates until all the values in the middle swath of P1 are in C2, and vice versa.
        Steps:
        1: Iterates through middle swath to check which values are already in each child
        2: If the value is not in the child, follows the mapping until an unfilled value is found
        3: Fills child with the correct mapped value

        input: 
        child (list): Child route to be inputted
        p1 (list): parent donor of the middle swath
        p2 (list): other parent route
        start (int): index at which the middle swath starts
        end (int): index at which the middle swath ends

        output:
        child (list): Inputted child
        '''
        for i in range(start, end + 1):
            if p2[i] not in child:
                value = p2[i]
                pos = i
                while child[pos] != -1:
                    value = p2[p1.index(value)]
                    pos = p1.index(value)
                child[pos] = p2[i]
        return child
    
    # Fill the remaining positions in Child 1
    c1 = fill_child(c1, p1, p2, start, end)
    # Fill the remaining positions in Child 2
    c2 = fill_child(c2, p2, p1, start, end)
    

    from_p2 = list(set(p2)- set(c1))
    from_p1 = list(set(p1) - set(c2))
    # Fill the rest of the positions with Parent 2's alleles for Child 1
    i = 0
    j =0
    while len(from_p2) >0:
        if -1 not in c1:
            break
        if c1[i] == -1:
            c1[i] = from_p2[0]
            from_p2.pop(0)
        i += 1
    
    # Fill the rest of the positions with Parent 1's alleles for Child 2
    while(len(from_p1)) >0:
        if -1 not in c2:
            break
        if c2[j] == -1:
            c2[j] = from_p1[0]
            from_p1.pop(0)
        j += 1

    # Checking child validity, if any isn't valid, substitute it by the parent
    if not individual_validator(c1):
        c1 = p1
    if not individual_validator(c2):
        c2 = p2

    return c1, c2

def tournament_selection(pop:list, pop_fit:list, ts=1):
    '''
    Tournament selection algorithm
    
    Inputs: pop (list) - List of individuals in the population
            pop_fit (list) - List of fitness values of the population given
        
    Outputs: list - selected individual
    '''
    # randomly selecting ts number of individuals from the population:
    # or, more specifically, choosing the individuals from the population via their index
    pool = random.choices([i for i in range(len(pop))], k=ts)

    # getting the pop_fit of the individuals of the given index
    pool_fits = [pop_fit[i] for i in pool]

    # finding out where in the pool fits the best fitness is
    best = np.argmax(pool_fits)

    # return the individual from the population whose index is the same as the index
    # in pool of the individual who was best in pool_fits
    return pop[pool[best]]

def SUS_selection(pop:list, pop_fit:list, n_sel = 1):
    '''
    Stochastic Universal Sampling (SUS) selection algorithm
    
    Inputs: pop (list) - List of individuals in the population
            pop_fit (list) - List of fitness values of the population given
        
    Outputs: list - selected individual
    '''
    fitness_sum = sum(pop_fit)

    # creating selection pointers
    pointer_distance = fitness_sum / n_sel # Distance between pointers
    start = random.uniform(0, pointer_distance) # Random start
    pointers = [start + i * pointer_distance for i in range(n_sel)]    # Evenly spaced
    
    current_fitness = 0
    index = 0
    
    for pointer in sorted(pointers):
        # Sum fitness values until the pointer threshold
        while current_fitness < pointer:
            current_fitness += pop_fit[index]
            index += 1
        # Add the corresponding individual to the list 
        # When only 1 is to be selected, return only the individual
        selected = pop[index - 1]
    
    return selected

def boltzmann_selection(pop, pop_fit, temperature=0.5):
    '''
    Boltzmann Selection algorithm for selecting individuals based on entropy
    
    Inputs: pop (list) - List of individuals in a population
            pop_fit (list) - List of fitness values of the population given
            temperature (float) - Temperature parameter (controls the amount of randomness)
        
    Output: list - Selected individual
    '''
    pop_fit = np.array(pop_fit)

    # Normalize fitness values to avoid overflow
    max_fit = np.max(pop_fit)
    if max_fit > 0:
        norm_fit = pop_fit / max_fit
    else:
        norm_fit = pop_fit
    
    # Exponential fitness values scaled by temperature
    scaled_fit = np.exp(norm_fit / max(temperature, 1e-10))  # Avoid division by zero or very small temperature
    
    # Boltzmann probabilities
    probabilities = scaled_fit / np.sum(scaled_fit)
    
    # Ensure probabilities do not contain NaN values
    if np.any(np.isnan(probabilities)):
        probabilities = np.ones_like(probabilities) / len(probabilities)
    
    selected_index = np.random.choice(len(pop), p=probabilities)
    
    return pop[selected_index]

def GA(initializer, evaluator, selector, crossover, mutator,
       pop_size, n_gens, p_xo, p_m, elite_func, geo_matrix,
       verbose=False, log_path=None, elitism=False, plot=False, seed=0):
    '''
    Run the Genetic Algorithm to solve a maximisation problem

    Inputs: initializer - Function to initialize the population
            evaluator - Function to evaluate the fitness of the population
            selector - Function for selection operation
            crossover - Function for crossover operation
            mutator - Function for mutation operation
            pop_size (int) - Size of the population
            n_gens (int) - Number of generations
            p_xo (float) - Probability of crossover
            p_m (float) - Probability of mutation
            elite_func - Function to determine elite individuals
            geo_matrix (DataFrame) - Geographic matrix for evaluations
            verbose (bool) - Whether to print verbose output
            log_path (str) - Path to log file
            elitism (bool) - Whether to use elitism
            seed (int) - Random seed for reproducibility
            plot (bool) - Whether to plot the results

    Outputs: pop, pop_fit - Final population and their fitness values.
    '''

    areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
    geo_matrix = pd.DataFrame(geo_matrix,index=areas, columns=areas)

    # 1. Setting up the seed:
    random.seed(seed)
    np.random.seed(seed)

    # Lists to store maximum and average fitness at each generation
    max_fitness_values = []
    avg_fitness_values = []

    # 2. Initializing the gen 0 population:
    population = initializer(pop_size)

    # 3. Evaluating the current population:
    pop_fit = evaluator(population)

    # 4. Main loop
    for gen in range(n_gens):

        # 4.1. Creating an empty offpsring population:
        offspring = []

        # 4.2. While the offspring population is not full:
        while len(offspring) < pop_size:
            # 4.2.1. Selecting the parents
            if selector == boltzmann_selection:
                temperature = max(0.1, 100 * (0.9 ** gen))
                p1 = selector(population, pop_fit, temperature)
                p2 = selector(population, pop_fit, temperature)
            else:
                p1 = selector(population, pop_fit)
                p2 = selector(population, pop_fit)

            # 4.2.2. Choosing between crossover and reproduction
            if random.random() < p_xo:
                if crossover == scx_xover:
                    o1, o2 = crossover(p1, p2, geo_matrix)
                else:
                    o1, o2 = crossover(p1, p2)
            else:
                o1, o2 = deepcopy(p1), deepcopy(p2)

            # 4.2.3. Mutating the offspring
            # Dynamic mutation rate (decrease over generations):
            dyn_p_m = p_m * (1 - gen / n_gens)

            for o in [o1, o2]:
                if random.random() < dyn_p_m:
                    if mutator == rgibnnm:
                        o = mutator(o, geo_matrix)
                    else:
                        o = mutator(o)

                if o not in offspring and o not in population:
                    offspring.append(o)

        # 4.3. Making sure offspring population doesnt exceed pop_size:
        offspring = offspring[:pop_size]

        # 4.4. If elitism is used, apply it:
        if elitism:
            elite, best_fit = elite_func(population, pop_fit)
            if isinstance(elite[0], list):
                offspring.extend(elite[:pop_size - len(offspring)])  # Ensure size limit
            else:
                offspring.append(elite)

        # 4.5. Replacing the current population with the offpsring population
        population = offspring

        # 4.6. Evaluating the current population
        pop_fit = evaluator(population)

        # 4.7. Displaying and logging the generation results = the best fits
        new_elite, new_fit = elite_func(population, pop_fit)

        # 4.8. Calculating maximum and average fitness
        max_fitness = max(pop_fit)
        avg_fitness = np.mean(pop_fit)

        max_fitness_values.append(max_fitness)
        avg_fitness_values.append(avg_fitness)

        if verbose:
            print(f'     {gen}       |       {new_elite[0]} - {new_fit[0]}       ')
            print('-' * 32)

        if log_path is not None:
            with open(log_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([seed, gen, new_fit, new_elite])

    # 5. Plotting if enabled
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_gens), max_fitness_values, label='Maximum Fitness')
        plt.plot(range(n_gens), avg_fitness_values, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution of Fitness')
        plt.legend()
        plt.grid(True)
        plt.show()


    return population, pop_fit

def generate_route(areas: list) -> list:
    '''
    Function that generates 1 random route by randomly selecting and removing items from the list of areas to visit
    Input:  areas (list of strings) - Possible areas to visit
    Output: individual (list of strings) - order in which the player visits the areas
    '''
    areas_copy = areas[1:]  # Not using 'D'
    random_route = ['D']  # Initialize the random route list with 'D' at the beginning
    
    while areas_copy:  # Until all areas are visited
        # Selecing an area, adding to the route and removing from the copy
        random_area = random.sample(areas_copy, 1)[0]
        random_route.append(random_area)
        areas_copy.remove(random_area)
    
    random_route.append('D')  # Add 'D' at the end of the random route
    return random_route

def generate_possible_route(areas:list) -> list:
    '''
    Function that generates 1 possible route by shuffing the list of areas to visit, then
    verifying if it is actually possible given the constraints
    Input:  areas (list of strings) - Possible areas to visit
    Output: individual (list of strings) - order in which the player visits the areas
    '''
    valitidy = False
    while not valitidy:
        route = generate_route(areas)
        valitidy = individual_validator(route)
    
    return route

def individual_fitness(route:list, geo_matrix:pd.DataFrame) -> int:
    '''
    Function  that evaluates an individual's fintess acoording to its
    Geo gain and the conditions given
    Input:  route (list of strings) - Order in which areas are visited
            geo_matrix (Dataframe) - Values of Geo gain
    Output: int - Fitness value of that individual
    '''
    if individual_validator(route):
        # In case the solution is valid
        if can_it_skip_KS(route):
            route_skip = deepcopy(route)
            route_skip.remove('KS')
            return max(route_total_geo(route, geo_matrix), route_total_geo(route_skip, geo_matrix))
        else:
            return route_total_geo(route, geo_matrix)
    else:
        # In case the solution is invalid
        return (route_total_geo(route, geo_matrix)-100)

def route_total_geo(route:list, geo_matrix:pd.DataFrame) -> int:
    '''
    Function that calculates the total Geo gain for a route
    Inputs: route (list of strings) - Order in which areas are visited
            geo_matrix (dataframe) - Geo gain values between each pair of areas
    Output: total_geo_gain - Sum of all values in the order given
    '''
    total_geo_gain = 0
    for i in range(len(route)-1):
        # Sum the value of the pair start-end in the matrix
        start = route[i]
        end = route[i + 1]
        total_geo_gain += geo_matrix.loc[start, end]

    return total_geo_gain

def create_population(areas:list) -> list:
    '''
    Creates a valid population
    Input: areas (list) - List of possible areas to visit
    Output: list - List of individuals (routes) of size pop_size
    '''
    def generate_pop(pop_size):

        return [generate_possible_route(areas) for _ in range(pop_size)]

    return generate_pop

def evaluate_population(geo_matrix):
    '''
    Returns a list of fitnesses
    '''
    
    def pop_evaluation(population):

        return [individual_fitness(route, geo_matrix) for route in population]

    return pop_evaluation

def get_n_elites_max(n):

    def get_elite(population, pop_fit):

        # getting the best n elites:

        bests_i = np.argsort(pop_fit)[-n:]

        # getting the fitnesses of the best n elites:

        return [population[i] for i in bests_i], [pop_fit[i] for i in bests_i]  # returning the list of elites and their list of fitnesses

    return get_elite

def generate_matrix(probability_positive_gain:int, areas:list) -> pd.DataFrame:
    '''
    Function that creates a randomly generated matrix of GEO gain
    Input: areas - list of possible areas to visit
        probability_positive_gain - probability that the GEO gain will be positive
        Note: This variable is influenced by luck and player skill
    Output: matrix - dataframe of generated GEO values
    '''
    # Our matrix will be in dataframe format for ease of use
    matrix = pd.DataFrame(index=areas, columns=areas)

    # Initial values
    for area1 in areas:
        for area2 in areas:
            if area1 == area2:
                matrix.loc[area1, area2] = 0  # No gain/loss when staying in the same area
            else:
                if random.random() <= probability_positive_gain:
                    matrix.loc[area1, area2] = round(random.uniform(0, 100), 2)  # Positive gain
                else:
                    matrix.loc[area1, area2] = round(random.uniform(-100, 0), 2)  # Negative gain

    # Rule: Geo gain from G to FC must be at least 3.2% less than the positive minimum gain
    lowest_gain = matrix[matrix > 0].min().min()
    min_geo_g_fc = lowest_gain * 0.968
    while matrix.loc['G', 'FC'] >= min_geo_g_fc:
        matrix.loc['G', 'FC'] = round(random.uniform(-100, 100), 2)

    return matrix

def create_matrixes_file():
    areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
    matrixes = [generate_matrix(0.8, areas) for i in range(15)]

    #Writing those matrixes into an excel file
    with ExcelWriter('matrixes.xlsx', engine='openpyxl') as writer:
        # Write each matrix to a separate sheet
        for i, matrix in enumerate(matrixes):
            sheet_name = f'{i+1}'
            matrix.to_excel(writer, sheet_name=sheet_name, index=True)


# Experiment
def run_experiment(mutator, crossover, selector):
    '''
    Run the Genetic Algorithm with standard parameters except for operators
    Inputs: crossover, mutator, selector (functions) - Genetic operators to test
    Outputs: pop_fit (list) - List of fitnesses to plot
    '''
    areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
    geo_matrix = generate_matrix(0.8, areas)

    # Run the GA with specified parameters
    population, pop_fit = GA(
        initializer=create_population(areas),
        evaluator=evaluate_population(geo_matrix),
        selector=selector,
        crossover=crossover,
        mutator=mutator,
        pop_size=50,
        n_gens=100,
        p_xo=0.7,
        p_m=0.1,
        elite_func=get_n_elites_max(2),
        geo_matrix=geo_matrix,
        verbose=False,
        log_path=None,
        elitism=True,
        seed=0,
        plot=False
    )
    return pop_fit  # Return fitness values for plotting

def compare_mutators(mutators, crossover, selector):
    '''
    Compare different mutator methods while keeping crossover and selector constant.

    Inputs: mutators (list) - List of mutator functions to compare
            crossover - Crossover function to keep constant
            selector - Selector function to keep constant

    Output: Dictionary with mutator names as keys and fitness values as values
    '''

    results = {}
    for mutator in mutators:
        key = mutator.__name__
        print(f'Running GA with mutator: {key}')
        pop_fit = run_experiment(mutator, crossover, selector)
        results[key] = pop_fit
    return results

def compare_selectors(mutator, crossover, selectors):
    '''
    Compare different selector methods while keeping crossover and mutator constant.

    Inputs: mutator - Mutator function to keep constant
            crossover - Crossover function to keep constant
            selectors (list) - List of selector functions to compare

    Output: Dictionary with mutator names as keys and fitness values as values
    '''

    results = {}
    for selector in selectors:
        key = selector.__name__
        print(f'Running GA with selector: {key}')
        pop_fit = run_experiment(mutator, crossover, selector)
        results[key] = pop_fit
    return results

def compare_crossovers(mutator, crossovers, selector):
    '''
    Compare different selector methods while keeping crossover and mutator constant.

    Inputs: mutator - Mutator function to keep constant
            crossovers (list) - List of crossover functions to compare
            selector - Selector function to keep constant

    Output: Dictionary with mutator names as keys and fitness values as values
    '''

    results = {}
    for crossover in crossovers:
        key = crossover.__name__
        print(f'Running GA with crossover: {key}')
        pop_fit = run_experiment(mutator, crossover, selector)
        results[key] = pop_fit
    return results

def plot_results(results, title):
    """
    Plot the results of the fitness values for different methods.

    Inputs: results (dict) - Dictionary with method names as keys and fitness values as values.
            title (str) - Title for the plot.
    """
    plt.figure(figsize=(12, 8))
    for label, fitness in results.items():
        plt.plot(fitness, label=label)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title(title)
    plt.legend()
    plt.show()
    return None