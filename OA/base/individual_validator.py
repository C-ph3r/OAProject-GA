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