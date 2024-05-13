import pandas as pd
from individual_validator import individual_validator

def route_total_geo(route:list, geo_matrix:pd.DataFrame) -> int:
    '''
    Function that calculates the total Geo gain for a route
    Inputs: route (list of strings) - Order in which areas are visited
            geo_matrix (dataframe) - Geo gain values between each pair of areas
    Output: total_geo_gain - Sum of all values in the order given
    '''
    total_geo_gain = 0
    for i in range(len(route)-1):
        # Sum the value of the pair start-end in the matrix
        start = route[i]
        end = route[i + 1]
        total_geo_gain += geo_matrix.loc[start, end]

    # If the last area is 'KS', then it was not visited and should not be added
    if route[-1] == 'KS':
        total_geo_gain -= geo_matrix.loc['D', 'KS']

    return total_geo_gain


def individual_fitness(route:list, geo_matrix:pd.DataFrame) -> int:
    '''
    Function  that evaluates an individual's fintess acoording to its
    Geo gain and the conditions given
    Input:  route (list of strings) - Order in which areas are visited
            geo_matrix (Dataframe) - Values of Geo gain
    Output: int - Fitness value of that individual
    '''
    if individual_validator(route):
        # In case the solution is valid
        return individual_fitness(route, geo_matrix)
    else:
        # In case the solution is invalid
        return -1