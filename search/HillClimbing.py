import random

class HillClimbing:
    def __init__(self, initial_plan, cost_function, generate_neighbors, max_iters=1000, delta_fraction=0.01):
        # Current best solution and its score
        self.current = initial_plan
        self.score = cost_function(initial_plan)

        # Store references for use during search
        self.cost_function = cost_function
        self.generate_neighbors = generate_neighbors
        self.max_iters = max_iters
        self.delta_fraction = delta_fraction

    def run(self):
        for iteration in range(self.max_iters):
            delta = self.delta_fraction * self.current['total_salary']
            neighbors = self.generate_neighbors(self.current, delta)
            best_neighbor = None
            best_score = self.score
            for neighbor in neighbors:
                s = self.cost_function(neighbor)
                if s > best_score:
                    best_score = s
                    best_neighbor = neighbor
            if best_neighbor:
                self.current = best_neighbor
                self.score = best_score
            else:
                break
        return self.current, self.score
