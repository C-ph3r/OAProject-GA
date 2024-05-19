import sys
sys.path.insert(0, '..')
from base.individual_validator import individual_validator
import numpy as np
import random

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

