import pandas as pd

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


def individual_validator(route:list) -> bool:
    '''
    Function that returns True if the individual is valid, given the conditions
    Input:  route (list of strings) - Order in which areas are visited
    Output: bool - Whether or not the route is valid
    '''
    # Condition 1: All sessions must begin in Dirtmouth (D)
    if route[0] != 'D':
        return False
    
    # Condition 2: City Storerooms (CS) cannot be visited right after Queen's Gardens (QG)
    if 'QG' in route and 'CS' in route[route.index('QG')+1:]:
        return False
    
    # Condition 3: Resting Grounds (RG) can only be reached in the last half of the session.
    if 'RG' in route[:len(route)//2]:
        return False
    
    # Condition 4: A session must end in 'D'. Or in ['D', 'KS'] only if 'DV' is visited right after 'QS'
    if 'QS' in route:
        index_qs = route.index('QS')
        if index_qs < len(route) - 1 and route[index_qs + 1] == 'DV':
            # Exception case: route can end in ['D', 'KS']
            if route[-2:] != ['D', 'KS'] and route[-1] != 'D':
                return False
        else:
            # Normal case: route must end in 'D'
            if route[-1] != 'D':
                return False
    
    # If no conditions apply, then the route is valid
    return True


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