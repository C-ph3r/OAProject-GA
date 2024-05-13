'''
Rules for GEO values:
• When considering different Geo earning values, know that more often than not
moving from one area to the next results in a Geo gain and not loss.
• When trying different Geo earning values, keep in mind that the Geo gained by
passing from Greenpath (G) to Forgotten Crossroads (FC) must be at least 3.2 %
less than the minimum between all the other positive Geo gains.
'''

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
    Output: matrix - dataframe of generated GEO values
    '''
    
    # Our matrix will be in dataframe format for ease of use
    matrix = pd.DataFrame(index=areas, columns=areas)
    
    # Initial values
    for area1 in areas:
        for area2 in areas:
            if area1 == area2:
                matrix.loc[area1, area2] = 0  # No gain/loss when staying in the same area
            else:
                if random.random() <= probability_positive_gain:
                    matrix.loc[area1, area2] = round(random.uniform(0, 100), 2)  # Positive gain
                else:
                    matrix.loc[area1, area2] = round(random.uniform(-100, 0), 2)  # Negative gain
    
    # Rule: Geo gain from G to FC must be at least 3.2% less than the positive minimum gain
    lowest_gain = matrix[matrix > 0].min().min()
    min_geo_g_fc = lowest_gain * 0.968
    while matrix.loc['G', 'FC'] >= min_geo_g_fc:
        matrix.loc['G', 'FC'] = round(random.uniform(-100, 100), 2)
    
    return matrix

# Testing
areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']
matrix = generate_matrix(0.5, areas)
print(matrix)