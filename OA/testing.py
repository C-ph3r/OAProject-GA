import random

def two_point_xover(p1, p2):

    # choosing xover points
    xp1 = random.randint(1, len(p1) - 2)
    xp2 = random.randint(xp1 + 1, len(p1) - 1)

    # generating the offspring
    o1 = p1[:xp1] + p2[xp1:xp2] + p1[xp2:]
    o2 = p2[:xp1] + p1[xp1:xp2] + p2[xp2:]

    return o1, o2

p1 = '000000'
p2 = '111111'

print(two_point_xover(p1, p2))