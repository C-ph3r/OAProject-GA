import random

def single_bit_mutation(individual, p_m):

    mutated_individual = []

    for bit in individual:

        if random.random() < p_m:

            mutated_individual.append(1 - bit)

        else:
            mutated_individual.append(bit)


    return mutated_individual

    # or...
    # return [1-bit if random.random() < p_m else bit for bit in individual]
