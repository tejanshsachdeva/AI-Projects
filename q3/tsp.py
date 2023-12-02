import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Function to calculate the total distance of a route
def calculate_total_distance(route, distances):
    total_distance = 0
    num_cities = len(route)

    for i in range(num_cities - 1):
        total_distance += distances[route[i]][route[i + 1]]

    # Add distance from the last city back to the starting city
    total_distance += distances[route[-1]][route[0]]

    return total_distance

# Function to initialize a population of routes
def initialize_population(num_individuals, num_cities):
    population = [np.random.permutation(num_cities).tolist() for _ in range(num_individuals)]
    return population

# Function to perform tournament selection
def select_tournament_indices(population, tournament_size):
    return np.random.choice(len(population), tournament_size, replace=False)

def create_tournament(population, indices):
    return [population[i] for i in indices]

def calculate_fitness(tournament, distances):
    return [calculate_total_distance(route, distances) for route in tournament]

def select_winner(tournament, fitness):
    return tournament[np.argmin(fitness)]

def tournament_selection(population, distances, tournament_size):
    indices = select_tournament_indices(population, tournament_size)
    tournament = create_tournament(population, indices)
    fitness = calculate_fitness(tournament, distances)
    return select_winner(tournament, fitness)

# Function to perform ordered crossover
def ordered_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)

    # Copy the segment between start and end from parent1 to the child
    child[start:end] = parent1[start:end]

    # Fill in the remaining positions with genes from parent2
    for i in range(len(parent2)):
        if parent2[i] not in child:
            for j in range(len(child)):
                if child[j] is None:
                    child[j] = parent2[i]
                    break

    return child

# Function to perform mutation (swap two random cities)
def select_indices(route):
    return np.random.choice(len(route), 2, replace=False)

def swap_elements(route, idx1, idx2):
    route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def mutation(route):
    idx1, idx2 = select_indices(route)
    return swap_elements(route, idx1, idx2)

# Function to plot the improvement over generations
def plot_improvement(improvements):
    sns.lineplot(x=range(len(improvements)), y=improvements, marker='o')
    plt.title('Improvement Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.show()

# Function to plot the city map with x and y axes
def plot_city_map(best_route, city_positions):
    city_positions = np.array(city_positions)
    x, y = city_positions[:, 0], city_positions[:, 1]

    best_route_positions = city_positions[best_route]
    best_route_x, best_route_y = best_route_positions[:, 0], best_route_positions[:, 1]

    sns.scatterplot(x=x, y=y, marker='o', label='Cities')
    plt.plot(np.append(best_route_x, best_route_x[0]), np.append(best_route_y, best_route_y[0]), linestyle='-', marker='o', color='b')
    plt.title('Best Route')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()


# Genetic algorithm function
def genetic_algorithm(num_generations, population_size, num_cities, distances, tournament_size):
    population = initialize_population(population_size, num_cities)
    improvements = []  # To store the best distance at each generation

    for generation in range(num_generations):
        new_population = []

        best_individual = min(population, key=lambda x: calculate_total_distance(x, distances))
        improvements.append(calculate_total_distance(best_individual, distances))

        while len(new_population) < population_size:
            # Select parents using tournament selection
            parent1 = tournament_selection(population, distances, tournament_size)
            parent2 = tournament_selection(population, distances, tournament_size)

            # Perform crossover
            child = ordered_crossover(parent1, parent2)

            # Perform mutation
            if random.random() < mutation_rate:
                child = mutation(child)

            new_population.append(child)

        population = new_population

    # Calculate total distances for all solutions in the population
    total_distances = [calculate_total_distance(x, distances) for x in population]

    # Find the index of the best solution
    best_index = np.argmin(total_distances)

    # Get the best solution and its total distance
    best_solution = population[best_index]
    best_distance = total_distances[best_index]

    # Append the total distance of the best solution to the improvements list
    improvements.append(best_distance)

    # Plotting the improvement over generations
    plot_improvement(improvements)

    return best_solution, calculate_total_distance(best_solution, distances)

# Example usage with logging and plotting
num_cities = 20
city_distances = np.random.randint(1, 100, size=(num_cities, num_cities))
np.fill_diagonal(city_distances, 0)

city_positions = np.random.rand(num_cities, 2)  # Random positions for city map visualization

population_size = 500
num_generations = 150
tournament_size = 5
mutation_rate = 0.1

best_route, best_distance = genetic_algorithm(
    num_generations, population_size, num_cities, city_distances, tournament_size
)

print("Best Route:", best_route)
print("Best Distance:", best_distance)

# Plot the city map
plot_city_map(best_route, city_positions)