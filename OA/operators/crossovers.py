import random


def one_point_xover(p1, p2):

    # choosing a crossover point
    xover_point = random.randint(1, len(p1) - 1)

    # generating the offspring
    o1 = p1[:xover_point] + p2[xover_point:]
    o2 = p2[:xover_point] + p1[xover_point:]

    return o1, o2
