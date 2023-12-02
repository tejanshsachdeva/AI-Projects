This Python script implements a Genetic Algorithm for solving the Traveling Salesman Problem (TSP). Here's a breakdown of the code:

1. **calculate_total_distance Function:**
   - Calculates the total distance of a given route based on a distance matrix.

2. **initialize_population Function:**
   - Initializes a population of routes, where each route is a permutation of cities.

3. **select_tournament_indices, create_tournament, calculate_fitness, and tournament_selection Functions:**
   - Implement tournament selection to choose parents for crossover.

4. **ordered_crossover Function:**
   - Performs ordered crossover, a type of crossover operation used in genetic algorithms.

5. **select_indices, swap_elements, and mutation Functions:**
   - Implement mutation by swapping two random cities in a route.

6. **plot_improvement and plot_city_map Functions:**
   - Provide plotting functions to visualize the improvement over generations and the best route on a city map.

7. **genetic_algorithm Function:**
   - Executes the genetic algorithm by iterating through generations, selecting parents, performing crossover and mutation, and creating a new population.

8. **Example Usage:**
   - Randomly generates a distance matrix representing the distances between cities.
   - Randomly generates positions for city map visualization.
   - Calls the `genetic_algorithm` function with specified parameters.
   - Prints the best route and its total distance.
   - Plots the improvement over generations and the city map with the best route.

The genetic algorithm is used to evolve a population of routes over generations, with the aim of finding the optimal route (shortest total distance) for visiting all cities. The provided example demonstrates the algorithm's application to a TSP with 20 cities, using random distance matrices and city positions.