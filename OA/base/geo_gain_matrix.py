import random
import pandas as pd
import numpy as np

def generate_matrix(probability_positive_gain:int, areas:list) -> pd.DataFrame:
    '''
    Function that creates a randomly generated matrix of GEO gain,
    based on the rules above
    Input: areas - list of possible areas to visit
        probability_positive_gain - probability that the GEO gain will be positive
        Note: This variable is influenced by luck and player skill
    Output: matrix - list of lists of generated GEO values
    '''
    
    matrix = []
    
    # Initial values
    for area1 in areas:
        row = []
        for area2 in areas:
            if area1 == area2:
                row.append(0)  # No gain/loss when staying in the same area
            else:
                if random.random() <= probability_positive_gain:
                    row.append(round(random.uniform(0, 100), 2))  # Positive gain
                else:
                    row.append(round(random.uniform(-100, 0), 2))  # Negative gain
        matrix.append(row)
    
    # Rule: Geo gain from G to FC must be at least 3.2% less than the positive minimum gain
    lowest_gain = min([min(filter(lambda x: x > 0, row)) for row in matrix])
    min_geo_g_fc = lowest_gain * 0.968

    for i, row in enumerate(matrix):
        if row[areas.index('FC')] >= min_geo_g_fc:
            matrix[i][areas.index('FC')] = round(random.uniform(-100, 100), 2)
    
    return matrix
