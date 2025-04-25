import queue
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Node Class 
# ------------------------------
class SpendingNode:
    def __init__(self, state, parent=None, action=None, path_cost=0, depth=0):
        """
        Initialize a node for the spending allocation search.
        
        Parameters:
            - state: Dictionary of spending allocations by category
            - parent: Parent node
            - action: Action that led to this state (category and amount)
            - path_cost: Cost to reach this node (g value)
            - depth: Depth in the search tree
        """
        self.state = state  # Dictionary of spending allocations
        self.parent = parent
        self.action = action
        self.g = path_cost  # g(n) value
        self.depth = depth
        self.h = 0  # Will be set during evaluation
        self.f = 0  # Will be set during evaluation
        self.id = id(self)  # Unique identifier for the node
    
    def get_path(self):
        """Return the path from root to this node."""
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))

# ------------------------------
# 2. Problem Definition
# ------------------------------
class SpendingAllocationProblem:
    def __init__(self, salary, categories, priorities, fixed_expenses=None, savings_priority=None):
        """
        Initialize the spending allocation problem.
        
        Parameters:
            - salary: Total monthly salary
            - categories: List of spending categories
            - priorities: Dictionary mapping categories to priority values
            - fixed_expenses: Dictionary of non-reducible expenses
            - savings_priority: Priority value for savings
        """
        self.salary = salary
        self.categories = categories
        self.priorities = priorities
        self.fixed_expenses = fixed_expenses or {}
        self.available_amount = salary - sum(self.fixed_expenses.values())
        self.max_priority = max(priorities.values()) if priorities else 0
        self.savings_priority = savings_priority or self.calculate_savings_priority()
        self.initial_state = {category: 0 for category in categories}
        for category, amount in self.fixed_expenses.items():
            if category in self.initial_state:
                self.initial_state[category] = amount
        
        # Step size for allocations (can be tuned)
        self.step_size = max(100, int(salary * 0.01))  # 1% of salary or minimum 100
        
    def calculate_savings_priority(self):
        """Calculate saving priority based on formula if not provided."""
        # Using formula from the project: P_saving = 5 - (Salary/100000) + 0.5 * Children + (1-H) + 0.5 * Car
        # Simplified version for now
        return 5 - (self.salary / 100000)
    
    def compute_normalization_factor(self, state):
        """Compute normalization factor 'a' for ideal spending percentages."""
        sum_terms = sum((self.max_priority - self.priorities[cat] + 1) for cat in self.categories)
        savings_term = self.max_priority - self.savings_priority + 1
        return 1 / (savings_term + sum_terms)
    
    def deviation(self, amount, category, state):
        """Calculate deviation from ideal spending for a category."""
        a = self.compute_normalization_factor(state)
        ideal = a * (self.max_priority - self.priorities[category] + 1)
        return abs((amount / self.salary) - ideal)
    
    def category_cost(self, amount, category, state):
        """Calculate cost for a specific category's spending."""
        if amount == 0:
            return 0
        a = self.compute_normalization_factor(state)
        return amount * self.deviation(amount, category, state) * self.priorities[category]
    
    def total_cost(self, state):
        """Calculate total cost of a spending plan."""
        total_spent = sum(state.values())
        savings = self.salary - total_spent
        
        # Sum of costs for all categories
        category_costs = sum(self.category_cost(state[cat], cat, state) for cat in self.categories)
        
        # Add savings cost
        savings_cost = savings * self.savings_priority
        
        return category_costs + savings_cost
    
    def heuristic(self, state):
        """Estimate cost to reach an optimal solution from current state."""
        # Remaining categories that aren't fully allocated yet
        unallocated_cats = [cat for cat in self.categories 
                           if cat not in self.fixed_expenses and state[cat] < self.available_amount]
        
        if not unallocated_cats:
            return 0
        
        # Simple heuristic: average deviation across remaining categories
        a = self.compute_normalization_factor(state)
        total_deviation = 0
        
        for cat in unallocated_cats:
            ideal_amount = a * (self.max_priority - self.priorities[cat] + 1) * self.salary
            current_deviation = abs(state[cat] - ideal_amount)
            total_deviation += current_deviation * self.priorities[cat]
            
        # Scale by number of categories to keep it admissible
        return total_deviation / len(unallocated_cats)
    
    def is_goal(self, state):
        """Check if state is a complete valid allocation."""
        total_spent = sum(state.values())
        # Within 1% of total salary is considered a complete allocation
        return abs(total_spent - self.salary) <= (self.salary * 0.01)
    
    def get_valid_actions(self, state):
        """Get all valid actions (category allocations) from current state."""
        actions = []
        total_spent = sum(state.values())
        remaining = self.salary - total_spent
        
        if remaining <= 0:
            return []
            
        step = min(self.step_size, remaining)
        
        for category in self.categories:
            # Skip fixed expense categories
            if category in self.fixed_expenses:
                continue
                
            # Add action to increase spending in this category
            actions.append((category, step))
            
        return actions
    
    def expand_node(self, node):
        """Generate child nodes from the current node."""
        children = []
        actions = self.get_valid_actions(node.state)
        
        for category, amount in actions:
            # Create a new state by adding the amount to the category
            new_state = node.state.copy()
            new_state[category] += amount
            
            # Calculate path cost (g value)
            path_cost = self.total_cost(new_state)
            
            # Create a new node
            child = SpendingNode(
                state=new_state,
                parent=node,
                action=(category, amount),
                path_cost=path_cost,
                depth=node.depth + 1
            )
            
            # Calculate heuristic (h value)
            child.h = self.heuristic(new_state)
            
            # Calculate f value
            child.f = child.g + child.h
            
            children.append(child)
            
        return children
    
    def print_node(self, node):
        """Print node information for debugging."""
        print(f"Depth: {node.depth}, Actions so far: {node.action}")
        print(f"Current state: {node.state}")
        print(f"Total spent: {sum(node.state.values())}/{self.salary}")
        print(f"f(n) = g(n) + h(n) = {node.g} + {node.h} = {node.f}")
        print("-" * 40)

# ------------------------------
# 3. A* Search Implementation with Separate Score Queue
# ------------------------------
class SpendingSearch:
    def __init__(self, problem):
        """
        Initialize the search process with a spending problem instance.
        
        Parameters:
            - problem: An instance of SpendingAllocationProblem
        """
        self.problem = problem
        
    def search(self, max_iterations=1000):
        """
        Execute A* search to find optimal spending allocation.
        
        Parameters:
            - max_iterations: Maximum number of nodes to expand
            
        Returns:
            - A SpendingNode instance representing the goal if found, otherwise None
        """
        # Initialize data structures
        # 1. Regular frontier (just nodes)
        frontier = {}  # Dictionary to store nodes: {node_id: node}
        # 2. Score queue (min heap priority queue for scores)
        score_queue = []  # Min heap for scores with node_id
        
        explored = set()
        
        # Create initial node
        initial_node = SpendingNode(self.problem.initial_state.copy())
        initial_node.g = self.problem.total_cost(initial_node.state)
        initial_node.h = self.problem.heuristic(initial_node.state)
        initial_node.f = initial_node.g + initial_node.h
        
        # Add to frontier and score queue
        frontier[initial_node.id] = initial_node
        heapq.heappush(score_queue, (initial_node.f, initial_node.id))
        
        iterations = 0
        
        print("Starting A* search with separate score queue...")
        
        while score_queue and iterations < max_iterations:
            iterations += 1
            
            # Get the node with lowest f score from score queue
            score, node_id = heapq.heappop(score_queue)
            
            # Get the corresponding node from frontier
            node = frontier.pop(node_id)
            
            # Print node information (can be commented out)
            if iterations % 50 == 0:  # Only print every 50th node to reduce output
                print(f"Iteration {iterations}:")
                self.problem.print_node(node)
            
            # Check if goal
            if self.problem.is_goal(node.state):
                print("Success! Found optimal allocation after", iterations, "iterations")
                return node
            
            # Add to explored
            # Convert state dict to tuple for hashing
            state_tuple = tuple(sorted(node.state.items()))
            explored.add(state_tuple)
            
            # Expand node
            for child in self.problem.expand_node(node):
                # Convert child state to tuple for hashing
                child_state_tuple = tuple(sorted(child.state.items()))
                
                if child_state_tuple not in explored:
                    # Check if a node with this state is already in frontier
                    duplicate = False
                    for existing_id, existing_node in frontier.items():
                        if tuple(sorted(existing_node.state.items())) == child_state_tuple:
                            # If existing node has worse score, replace it
                            if existing_node.f > child.f:
                                # Remove from score queue (will be marked as removed)
                                # We'll add a new entry instead
                                frontier[existing_id] = child
                                heapq.heappush(score_queue, (child.f, existing_id))
                            duplicate = True
                            break
                    
                    if not duplicate:
                        # Add to frontier and score queue
                        frontier[child.id] = child
                        heapq.heappush(score_queue, (child.f, child.id))
                
        print(f"Search terminated after {iterations} iterations without finding a goal")
        if iterations >= max_iterations:
            print("Search reached maximum iterations limit")
        return None

# ------------------------------
# 4. Visualization Functions
# ------------------------------
def visualize_spending_plan(solution_node, problem):
    """
    Create visualizations for the optimal spending plan.
    
    Parameters:
        - solution_node: The goal node with optimal allocations
        - problem: The problem instance
    """
    if not solution_node:
        print("No solution to visualize")
        return
    
    # Extract the spending plan
    spending_plan = solution_node.state
    
    # Calculate percentages
    total = sum(spending_plan.values())
    percentages = {cat: (amount/total)*100 for cat, amount in spending_plan.items()}
    
    # Prepare data for plotting
    categories = list(spending_plan.keys())
    amounts = [spending_plan[cat] for cat in categories]
    
    # Create a pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(amounts, labels=categories, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Optimal Spending Allocation')
    plt.show()
    
    # Create a bar chart showing allocation vs priority
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(categories))
    width = 0.35
    
    # Normalized amounts (as percentage of salary)
    normalized_amounts = [(amount/problem.salary)*100 for amount in amounts]
    
    # Normalized priorities (as percentage of max priority)
    priorities = [problem.priorities[cat] for cat in categories]
    normalized_priorities = [(p/problem.max_priority)*100 for p in priorities]
    
    plt.bar(indices - width/2, normalized_amounts, width, label='Allocation %')
    plt.bar(indices + width/2, normalized_priorities, width, label='Priority %')
    
    plt.xlabel('Categories')
    plt.ylabel('Percentage')
    plt.title('Spending Allocation vs Priority')
    plt.xticks(indices, categories, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print detailed allocation plan
    print("=== DETAILED SPENDING PLAN ===")
    print(f"Total Salary: {problem.salary} DZD")
    print("Category Allocations:")
    for cat in categories:
        print(f"  {cat}: {spending_plan[cat]} DZD ({percentages[cat]:.1f}%) - Priority: {problem.priorities[cat]}")
    
    print(f"Total Allocated: {total} DZD")
    print(f"Savings: {problem.salary - total} DZD ({((problem.salary - total)/problem.salary)*100:.1f}%)")
    print(f"Final Score (Total Cost): {solution_node.g}")
    print(f"Path Length: {solution_node.depth}")

# ------------------------------
# 5. Example Usage
# ------------------------------
def run_example():
    """Run an example search with sample data."""
    # Sample data
    salary = 120000  # 120,000 DZD monthly salary
    categories = ["Housing", "Food", "Transportation", "Healthcare", "Education", "Entertainment"]
    priorities = {
        "Housing": 5,
        "Food": 4,
        "Transportation": 3,
        "Healthcare": 5,
        "Education": 4,
        "Entertainment": 2
    }
    fixed_expenses = {
        "Housing": 30000,  # Fixed rent
        "Healthcare": 5000  # Fixed health insurance
    }
    
    # Create problem instance
    problem = SpendingAllocationProblem(
        salary=salary,
        categories=categories,
        priorities=priorities,
        fixed_expenses=fixed_expenses
    )
    
    # Create search instance
    search = SpendingSearch(problem)
    
    # Run search
    solution = search.search(max_iterations=500)
    
    # Visualize results
    if solution:
        visualize_spending_plan(solution, problem)
    else:
        print("No solution found")

# Run the example
# Uncomment to run:
run_example()