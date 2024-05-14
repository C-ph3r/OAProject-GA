'''3.4 The proposed IRGIBNNM
We propose a hybrid mutation called: IRGIBNNM.
In this mutation we combine two mutation operators, the inversion mutation and
RGIBNNM.
The IRGIBNNM initially applies the inversion mutation on an individual, and
then the RGIBNNM is applied to the resulting individual. Thus, the new offspring
benefit from both mutations’ characteristics, attempting to enhance the
performance of both mutations, by increasing diversity in the search space, and
therefore to provide better results. The IRGIBNNM is depicted by Example (4).

Consider the following route (C):
C= (5 3 10 9 8 1 2 7 4) with cost =19, as depicted in Figure (2). 

To apply IRGIBNNM:
1. Select two random genes, e.g. the third and eighth genes.
2. A= Inversion Mutation(C). The resulting offspring
A = (5 3 10 2 1 8 9 7 4) with cost = 18.2 (see Figure 2).
3. Apply RGIBNNM(A) as follows:
• Select a random gene from A, e.g. the random gene is the eighth gene, i.e.
the random city is (7).
• Find the nearest city to the random city (7), which is city (3) in this case.
• Get a random city around city (3) in the range (± 5); e.g. city (9).
• Apply the Exchange mutation on chromosome A by swapping the cities 7
and 9, as shown in (Figure (3)). The final output offspring becomes:
Offspring = (5 3 10 2 1 8 7 9 4) with cost = (17.1).
'''

import random
import copy
from base.individual_validator import individual_validator



# Function to perform RGIBNNM
def rgibnnm(route):
    random_index = random.randint(0, len(route) - 1)
    random_city = route[random_index]
    nearest_city = min((c for c in route if c != random_city), key=lambda c: distances[random_city][c])
    range_of_cities = [c for c in route if abs(c - nearest_city) <= 5 and c != random_city]
    if range_of_cities:
        new_city = random.choice(range_of_cities)
        route[route.index(random_city)] = new_city
        route[route.index(new_city)] = random_city
    return route


# Step 1: Inversion Mutation
A = inversion_mutation(copy.deepcopy(route))
print("After Inversion Mutation: ", A)

# Step 2: RGIBNNM
offspring = rgibnnm(copy.deepcopy(A))
print("After RGIBNNM: ", offspring)
print("Cost: ", calculate_cost(offspring))

