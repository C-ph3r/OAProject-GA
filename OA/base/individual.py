import random

'''
A solution for this problem should be a vector of size 11, the number of
areas the player passes through in any given session.

Each index in the vector corresponds to the codename of the area the player visits
in order. If an area is after the final 'D', it is considered to not be visited.

Example solution: ['D', 'FC', 'G', 'QS', 'DV', 'QG', 'CS', 'SN', 'RG', 'D', 'KS']

Considering the following:
Dirtmouth (D) - Start and end
Forgotten Crossroads (FC)
Greenpath (G)
Queen's Station (QS)
Queen's Gardens (QG)
City Storerooms (CS)
King’s Station (KS)
Resting Grounds (RG)
Distant Village (DV)
Stag Nest (SN)


Rules:
• All sessions must begin and end in Dirtmouth (D).
• Your friend is ok with not visiting King’s Station (KS) as long as they visit the
Distant Village (DV) immediately after Queen’s Station (QS). They say that they
are ok with this skip because the QS-DV sequence is very hard and tiring, so the
addition of KS is unnecessary.
• For similar reasons, City Storerooms (CS) cannot be visited right after Queen’s
Garden’s (QG).
• The Resting Grounds (RG) can only be reached in the last half of the session.
'''

areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN']

def generate_random_solution(areas):
    return ['D'] + random.choices(areas, len(areas), replace=False)

def generate_possible_solution(areas):
    