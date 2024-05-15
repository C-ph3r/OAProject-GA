import random

def swap_mutation(tour):
    """
    Apply swap mutation to a tour in the TSP population.
    
    Parameters:
    - tour: List representing the tour (list of city indices)
    
    Returns:
    - mutated_tour: List representing the mutated tour
    """
    # Make a copy of the tour to avoid modifying the original
    mutated_tour = tour.deepcopy()
    
    # Select two distinct random indices in the tour
    idx1, idx2 = random.sample(range(len(mutated_tour)), 2)
    
    # Swap the cities at the selected indices
    mutated_tour[idx1], mutated_tour[idx2] = mutated_tour[idx2], mutated_tour[idx1]
    
    # Ensure that all cities are visited exactly once
    mutated_set = set(mutated_tour)
    for city in tour:
        if city not in mutated_set:
            mutated_tour.append(city)
            break
    
    return mutated_tour