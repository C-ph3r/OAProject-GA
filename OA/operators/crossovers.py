import numpy as np
from copy import deepcopy
import random

def order_xover(p1,p2):
    '''
    Performs order crossover on 2 parents. Steps:
    1- Select a randomly sized number of characters from the middle of both parents
    2- Assign the belt from parent 1 to child 1, and the belt from parent 2 to child 2
    3- Understand the characters that are already in each child, and those from the parents that are not yet in the child
    4- Assign the unused characters from parent 2 (in their original order), to child 1 and vice versa

    example input:
    Parent 1 = 1 2 3 | 4 5 6 7 | 8 9
    Parent 2 = 4 5 2 | 1 8 7 6 | 9 3
    
    example output:
    Children 1 = 2 1 8 | 4 5 6 7 | 9 3
    Children 2 = 3 4 5 | 1 8 7 6 | 9 2
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

    
def pmx_xover(p1,p2):
    '''
    Performes partially mapped crossover 
    '''
    c1 = p1
    c2 = p2

    #Choosing size of the middle belt
    size_middle_belt = random.randint(1, len(p1)//2 -1)
   

    #Assigning the middle belt to children
    c1[size_middle_belt: -size_middle_belt] = p2[size_middle_belt: -size_middle_belt]
    c2[size_middle_belt: -size_middle_belt] = p1[size_middle_belt: -size_middle_belt]

    #Acessing the values present in the belts
    belt_1 = p1[size_middle_belt: -size_middle_belt]
    belt_2 = p2[size_middle_belt: -size_middle_belt]

    for i in range(len(c1)):
        if c1[i] in belt_2:
            aux = belt_2.index(c1[i])
            c1[i] = belt_1[aux]

        if c2[i] in belt_1:
            aux = belt_1.index(c2[i])
            c2[i] = belt_2[aux]

    return c1,c2

t1 = ["8","4", "7", "3", "6", "2", "5", "1", "9", "0"]
t2 = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9"]


print(pmx_xover(t1,t2))