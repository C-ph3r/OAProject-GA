def individual_validator(route:list) -> bool:
    '''
    Function that returns True if the individual is valid, given the conditions
    Input:  route (list of strings) - Order in which areas are visited
    Output: bool - Whether or not the route is valid
    '''
    # Condition 1: All sessions must begin and end in Dirtmouth (D)
    if route[0] != 'D' or route[-1] != 'D':
        return False
    
    # Condition 2: City Storerooms (CS) cannot be visited right after Queen's Gardens (QG)
    if  route[route.index('QG')+1] == 'CS':
        return False
    
    # Condition 3: Resting Grounds (RG) can only be reached in the last half of the session.
    if 'RG' in route[:len(route)//2]:
        return False
    
    # If no conditions apply, then the route is valid
    return True


def can_it_skip_KS(route:list) -> bool:
    '''
    Function that returns True if this route meets the conditions to skip 'KS'
    Input: route (list of strings) - Order in which areas are visited
    Output: bool - Whether or not Ks can be skipped
    '''
    # If 'DV' is visited right after 'QS', the route may skil 'KS'
    index_qs = route.index('QS')
    if index_qs < len(route) - 1 and route[index_qs + 1] == 'DV':
        return True
    else:
        return False
    
