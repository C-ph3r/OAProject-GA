import numpy as np
import random

def order_xover(p1,p2):
    '''
    Performs order crossover on 2 parents. Steps:
    1- Select a randomly sized number of characters from the middle of both parents
    2- Assign the belt from parent 1 to child 1, and the belt from parent 2 to child 2
    3- Understand the characters that are already in each child, and those from the parents that are not yet in the child
    4- Assign the unused characters from parent 2 (in their original order), to child 1 and vice versa

    Parameters:
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
        if c1[i] == -1:
            c1[i] = from_2[0]
            from_2.pop(0)

    for i in range(len(c2)):
        if c2[i] == -1:
            c2[i] = from_1[0]
            from_1.pop(0)

    return c1,c2

def position_xover(p1,p2):
    '''
    Performs position crossover on two parent solutions
    Steps:
    1 - Chooses a random number of values to keep
    2 - Transfers said values in their original positions from p1 to c1 and p2 to c2
    3 - Transfers the remaining values crossed over from p2 to c1 and p1 to c2

    
    Parameters:
    p1 (list): first parent on which to perform crossover
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

    

    return c1,c2
