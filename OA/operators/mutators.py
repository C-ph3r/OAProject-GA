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


def buddy_bit_mutation(individual, p_m):

    mutated_individual = []

    i = 0

    while i < len(individual):

        if random.random() < p_m:
            if not i == len(individual) - 1:
                mutated_individual.extend([1 - individual[i], 1 - individual[i + 1]])
            else:
                mutated_individual.append(1 - individual[i])
                mutated_individual[0] = 1 - mutated_individual[0]

        else:
            if not i == len(individual) - 1:
                mutated_individual.extend([individual[i], individual[i + 1]])
            else:
                mutated_individual.append(individual[i])

        i += 1


def neighbourhood_mutation(hamming_dist):

    def mutate(individual, p_m):
        pass

    return mutate



def test_func():
    print("hello world")
    print("what")
    print("quero commitar")
    pass