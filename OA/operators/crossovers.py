import random


def one_point_xover(p1, p2):

    # choosing a crossover point
    xp = random.randint(1, len(p1) - 1)

    # generating the offspring
    o1 = p1[:xp] + p2[xp:]
    o2 = p2[:xp] + p1[xp:]

    return o1, o2

def two_point_xover(p1, p2):

    # choosing xover points
    xp1 = random.randint(1, len(p1) - 2)
    xp2 = random.randint(xp1 + 1, len(p1) - 1)

    # generating the offspring
    o1 = p1[:xp1] + p2[xp1:xp2] + p1[xp2:]
    o2 = p2[:xp1] + p1[xp1:xp2] + p2[xp2:]

    return o1, o2