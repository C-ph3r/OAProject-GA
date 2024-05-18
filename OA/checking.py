from operators.selection_algorithms import boltzmann_selection

population = ['A', 'B', 'C', 'D']
fitnesses = [10, 20, 30, 40]
initial_temperature = 100.0  # Higher initial temperature for more exploration
cooling_rate = 0.90
generations = 10

selected_individuals = []

for gen in range(generations):
    temperature = initial_temperature * (cooling_rate ** gen)
    selected_individual = boltzmann_selection(population, fitnesses, temperature)
    selected_individuals.append(selected_individual)
    print(f'Generation {gen + 1}: Selected individual: {selected_individual}, Temperature: {temperature:.2f}')

# Print the selected individuals over all generations
print(f'Selected individuals over {generations} generations: {selected_individuals}')