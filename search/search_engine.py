import queue
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from copy import deepcopy

# ------------------------------
# 1. Node Class Definition
# ------------------------------
class SpendingNode:
    def __init__(self, state, parent=None, level=0, g=0, h=0):
        """
        Initialize a spending plan node.
        
        Input Parameters:
            - state: Dictionary representing category-value pairs for the spending plan
                    Example: {'food': 900, 'education': 350, 'savings': 500, 'transport': 250}
            - parent: The parent SpendingNode that generated this node. Default is None.
            - level: The level in the tree (related to which category priority is being set)
            - g: The cost function value (deviation from optimal spending). Default is 0.
            - h: The heuristic value (estimate of future cost). Default is 0.
            
        Output:
            - A SpendingNode instance with attributes: state, parent, level, g, h, and f.
        """
        self.state = state.copy()
        self.parent = parent
        self.level = level
        self.g = g  # Actual cost from start to this node
        self.h = h  # Heuristic cost
        self.f = g + h  # Total evaluation function
    
    def __lt__(self, other):
        """
        Compare this node with another node based on the evaluation function (f).
        Used for priority queue ordering.
        """
        return self.f < other.f
    
    def __eq__(self, other):
        """
        Check equality with another SpendingNode based on the state.
        """
        return isinstance(other, SpendingNode) and self.state == other.state
    
    def __hash__(self):
        """
        Compute a hash value for the node.
        """
        return hash(tuple(sorted(self.state.items())))

# ------------------------------
# 2. SpendingAllocationProblem Class
# ------------------------------
class SpendingAllocationProblem:
    def __init__(self, salary, priorities, user_input_plan, num_children=0, 
                 has_car=False, has_house=False, fixed_categories=None, dataset=None):
        """
        Initialize the spending allocation problem.
        
        Input Parameters:
            - salary: User's monthly salary
            - priorities: Dictionary of categories with their priority values (lower is higher priority)
                         Example: {'food': 1, 'education': 2, 'savings': 3, 'transport': 4}
            - user_input_plan: User's initial spending plan
                              Example: {'food': 900, 'education': 350, 'savings': 500, 'transport': 250}
            - num_children: Number of children the user has
            - has_car: Boolean indicating if the user owns a car
            - has_house: Boolean indicating if the user owns a house
            - fixed_categories: Dictionary of categories with fixed spending amounts
            - dataset: Dataframe containing spending patterns data
            
        Output:
            - An instance of SpendingAllocationProblem with problem attributes initialized
        """
        self.salary = salary
        self.priorities = priorities
        self.categories = list(priorities.keys())
        self.fixed_categories = fixed_categories or {}
        self.user_input_plan = user_input_plan
        self.num_children = num_children
        self.has_car = has_car
        self.has_house = has_house
        self.dataset = dataset
        
        # Filter dataset rows according to user conditions
        self.filtered_dataset = self.filter_dataset()
        
        # Create initial state from user input plan
        self.initial_state = self.create_initial_state()
        
        # Calculate savings priority
        self.max_priority = max(priorities.values())
        self.savings_priority = self.calculate_savings_priority()
        
        # Sort categories by priority (lowest value = highest priority)
        self.sorted_categories = sorted(self.priorities.keys(), key=lambda x: self.priorities[x])
    
    def filter_dataset(self):
        """
        Filter the dataset based on user conditions (children, car, house).
        """
        if self.dataset is None:
            return []
            
        filtered = self.dataset.copy()
        
        # Filter by number of children if that column exists
        if 'children' in filtered.columns:
            filtered = filtered[filtered['children'] == self.num_children]
            
        # Filter by car ownership if that column exists
        if 'has_car' in filtered.columns:
            filtered = filtered[filtered['has_car'] == self.has_car]
            
        # Filter by house ownership if that column exists
        if 'has_house' in filtered.columns:
            filtered = filtered[filtered['has_house'] == self.has_house]
            
        return filtered
    
    def calculate_savings_priority(self):
        """
        Calculate savings priority based on salary and number of children.
        Higher value means savings is more important.
        """
        return (self.salary / 25000) - 0.75 * self.num_children
    
    def create_initial_state(self):
        """
        Create the initial user plan:
        - Fixed expenses are already set.
        - Other categories are taken from user input or default to 0.
        """
        state = {category: 0 for category in self.categories}
        
        # Fill in user-defined or fixed values
        for category in self.categories:
            if category in self.fixed_categories:
                state[category] = self.fixed_categories[category]
            elif category in self.user_input_plan:
                state[category] = self.user_input_plan[category]
        
        return state
    
    def is_goal_state(self, state):
        """
        Check if a state is a goal state:
        - Total spending is within +/- 5% of salary
        - Savings are at least 10% of salary
        - No category is underfunded below minimum threshold
        
        Input Parameters:
            - state: A dictionary of spending values for each category
            
        Output:
            - Boolean: True if the state meets all goal criteria, False otherwise
        """
        total_spending = sum(state.values())
        savings = self.salary - total_spending
        
        # Check if total spending is within +/- 5% of salary
        spending_in_range = (self.salary * 0.95) <= total_spending <= (self.salary * 1.05)
        
        # Check if savings are at least 10% of salary
        min_savings = 0.1 * self.salary  # 10% of salary
        sufficient_savings = savings >= min_savings
        
        # Check if any category is underfunded
        min_category_value = 1000  # Example minimum threshold, adjust as needed
        underfunded = any(state[cat] < min_category_value for cat in state 
                         if cat not in self.fixed_categories and cat != 'savings')
        
        return spending_in_range and sufficient_savings and not underfunded
    
    def get_heuristic(self, state):
        """
        Calculate heuristic value for a state.
        This estimates how far the state is from being optimal.
        
        Input Parameters:
            - state: A dictionary of spending values for each category
            
        Output:
            - A numerical value representing the heuristic estimate
        """
        total_spent = sum(state.values())
        savings = self.salary - total_spent
        
        # Penalty for deviation from salary
        salary_deviation = abs(total_spent - self.salary)
        
        # Penalty for insufficient savings
        min_savings = 0.1 * self.salary
        savings_penalty = max(0, min_savings - savings) * 2  # Multiply by 2 to give more weight
        
        # Penalty for priorities mismatch
        priority_penalty = 0
        for category, priority in self.priorities.items():
            # Lower priority values (higher actual priority) should have higher spending
            # Calculate ideal spending for each category based on priority
            if category not in self.fixed_categories:
                inverse_priority = self.max_priority - priority + 1
                ideal_percentage = inverse_priority / sum(self.max_priority - p + 1 
                                                         for p in self.priorities.values() 
                                                         if category not in self.fixed_categories
)
                                                        
                ideal_spending = ideal_percentage * (self.salary * 0.9)  # Assuming 10% savings
                priority_penalty += abs(state[category] - ideal_spending) * inverse_priority
        
        return salary_deviation + savings_penalty + priority_penalty / 100
    
    def get_cost(self, state):
        """
        Calculate the actual cost of a state.
        This measures how much the state deviates from the optimal solution.
        
        Input Parameters:
            - state: A dictionary of spending values for each category
            
        Output:
            - A numerical value representing the cost
        """
        total_spent = sum(state.values())
        savings = self.salary - total_spent
        
        # Cost for deviation from salary
        cost = abs(total_spent - self.salary)
        
        # Additional cost if savings are below minimum
        min_savings = 0.1 * self.salary
        if savings < min_savings:
            cost += (min_savings - savings) * (1 + self.savings_priority)
        
        # Cost for not respecting priorities
        for category, priority in self.priorities.items():
            if category not in self.fixed_categories:
                # Higher priority categories (lower value) should have less deviation
                inverse_priority = self.max_priority - priority + 1
                if category in state:
                    # If this is the savings category, use savings_priority
                    if category == 'savings':
                        weight = self.savings_priority
                    else:
                        weight = inverse_priority / self.max_priority
                    
                    # Calculate ideal value - simplified for demonstration
                    ideal_value = self.salary * (0.1 + 0.1 * weight)
                    cost += abs(state[category] - ideal_value) * weight
        
        return cost
    
    def expand_node(self, node):
        """
        Generate children nodes by exploring different values for the current priority category.
        
        Input Parameters:
            - node: The current SpendingNode to expand
            
        Output:
            - A list of child SpendingNode instances
        """
        # If we've set values for all categories, return empty list
        if node.level >= len(self.sorted_categories):
            return []
        
        current_category = self.sorted_categories[node.level]
        children = []
        
        # Skip fixed categories
        if current_category in self.fixed_categories:
            # Create a child with the fixed value and increment level
            child_state = node.state.copy()
            child_state[current_category] = self.fixed_categories[current_category]
            child_node = SpendingNode(
                state=child_state,
                parent=node,
                level=node.level + 1,
                g=self.get_cost(child_state),
                h=self.get_heuristic(child_state)
            )
            children.append(child_node)
            return children
        
        # Get unique values for current category from filtered dataset
        if not self.filtered_dataset.empty and current_category in self.filtered_dataset.columns:
            unique_values = set(self.filtered_dataset[current_category].unique())
        else:
            # If no data available, generate some reasonable values
            base_value = self.user_input_plan.get(current_category, self.salary * 0.1)
            unique_values = {
                max(100, base_value * 0.7),
                base_value,
                min(self.salary * 0.5, base_value * 1.3)
            }
        
        # Create child nodes for each possible value
        for value in unique_values:
            if current_category in self.fixed_categories and value != self.fixed_categories[current_category]:
                continue  # Skip if it's a fixed category with a different value
                
            child_state = node.state.copy()
            child_state[current_category] = value
            
            child_node = SpendingNode(
                state=child_state,
                parent=node,
                level=node.level + 1,
                g=self.get_cost(child_state),
                h=self.get_heuristic(child_state)
            )
            children.append(child_node)
        
        return children

# ------------------------------
# 3. A* Search Implementation
# ------------------------------
def astar_search(problem, max_iterations=1000):
    """
    Perform A* search to find an optimal spending plan.
    
    Input Parameters:
        - problem: An instance of SpendingAllocationProblem
        - max_iterations: Maximum number of iterations to prevent infinite loops
        
    Output:
        - A SpendingNode instance representing the goal state if found, otherwise None
    """
    # Create initial node
    initial_node = SpendingNode(
        state=problem.initial_state,
        level=0,
        g=problem.get_cost(problem.initial_state),
        h=problem.get_heuristic(problem.initial_state)
    )
    
    # Initialize priority queue
    frontier = queue.PriorityQueue()
    frontier.put(initial_node)
    
    # Keep track of explored states and states in frontier
    explored = set()
    frontier_states = {hash(initial_node)}
    
    iterations = 0
    while not frontier.empty() and iterations < max_iterations:
        iterations += 1
        
        # Get node with lowest f value
        current_node = frontier.get()
        frontier_states.remove(hash(current_node))
        
        # Print current node state for debugging
        print(f"Iteration {iterations}: Exploring node at level {current_node.level}")
        print(f"Current state: {current_node.state}")
        print(f"f = {current_node.f} (g = {current_node.g}, h = {current_node.h})")
        
        # Check if current node is goal state
        if problem.is_goal_state(current_node.state):
            print("\nGoal state found!")
            return current_node
        
        # Add to explored set
        state_hash = hash(current_node)
        explored.add(state_hash)
        
        # Expand current node
        children = problem.expand_node(current_node)
        print(f"Generated {len(children)} children nodes")
        
        # Process each child node
        for child in children:
            child_hash = hash(child)
            
            # Skip if already explored or in frontier with lower cost
            if child_hash in explored:
                continue
                
            if child_hash in frontier_states:
                # This is a simplification; in a real implementation, 
                # we would need to update the node in the frontier if this one has lower cost
                continue
            
            # Add child to frontier
            frontier.put(child)
            frontier_states.add(child_hash)
        
        print("-" * 50)
    
    print("\nNo solution found within iteration limit.")
    return None

# ------------------------------
# 4. Utility Functions
# ------------------------------
def reconstruct_path(node):
    """
    Reconstruct the path from the initial state to the goal.
    
    Input Parameters:
        - node: The goal SpendingNode
        
    Output:
        - A list of states representing the path
    """
    path = []
    current = node
    
    while current:
        path.append(current.state)
        current = current.parent
    
    path.reverse()  # Reverse to get path from start to goal
    return path

def plan_to_dataframe(plan):
    """
    Convert a spending plan to a DataFrame for visualization.
    
    Input Parameters:
        - plan: A dictionary of category-value pairs
        
    Output:
        - A pandas DataFrame
    """
    df = pd.DataFrame([plan])
    df = df.T.reset_index()
    df.columns = ['Category', 'Amount']
    return df

def optimize_user_plan(user_salary, user_priorities, user_plan, num_children=0,
                      has_car=False, has_house=False, fixed_categories=None, dataset=None):
    """
    Optimize a user's spending plan using A* search.
    
    Input Parameters:
        - user_salary: User's monthly salary
        - user_priorities: Dictionary of categories with priority values
        - user_plan: User's initial spending plan
        - num_children: Number of children
        - has_car: Boolean indicating car ownership
        - has_house: Boolean indicating house ownership
        - fixed_categories: Dictionary of categories with fixed values
        - dataset: DataFrame containing spending pattern data
        
    Output:
        - The optimized spending plan as a dictionary
    """
    # Create problem instance
    problem = SpendingAllocationProblem(
        salary=user_salary,
        priorities=user_priorities,
        user_input_plan=user_plan,
        num_children=num_children,
        has_car=has_car,
        has_house=has_house,
        fixed_categories=fixed_categories,
        dataset=dataset
    )
    
    # Run A* search
    goal_node = astar_search(problem)
    
    if goal_node:
        # Reconstruct path
        path = reconstruct_path(goal_node)
        
        # Return optimized plan
        return goal_node.state
    
    return user_plan  # Return original plan if no better plan found

# ------------------------------
# 5. Example Usage
# ------------------------------

# Create a sample dataset
def create_sample_dataset():
    """Create a sample dataset for demonstration purposes."""
    data = {
        'food': [800, 900, 1000, 1200, 1500],
        'housing': [5000, 6000, 7000, 8000, 9000],
        'transport': [200, 300, 400, 500, 600],
        'education': [300, 400, 500, 600, 700],
        'entertainment': [200, 300, 400, 500, 600],
        'savings': [1000, 1500, 2000, 2500, 3000],
        'children': [0, 1, 2, 0, 1],
        'has_car': [True, False, True, False, True],
        'has_house': [False, True, True, False, False]
    }
    return pd.DataFrame(data)

# Example usage
def example_usage():
    # Create sample dataset
    dataset = create_sample_dataset()
    
    # User inputs
    user_salary = 15000
    user_priorities = {
        'food': 1,
        'housing': 2,
        'transport': 4,
        'education': 3,
        'entertainment': 5,
        'savings': 6
    }
    user_plan = {
        'food': 1200,
        'housing': 7000,
        'transport': 400,
        'education': 500,
        'entertainment': 300,
        'savings': 2000
    }
    fixed_categories = {
        'housing': 7000  # Housing cost is fixed
    }
    
    # Optimize the plan
    optimized_plan = optimize_user_plan(
        user_salary=user_salary,
        user_priorities=user_priorities,
        user_plan=user_plan,
        num_children=1,
        has_car=True,
        has_house=True,
        fixed_categories=fixed_categories,
        dataset=dataset
    )
    
    # Display results
    print("\nUser's Original Plan:")
    print(plan_to_dataframe(user_plan))
    
    print("\nOptimized Plan:")
    print(plan_to_dataframe(optimized_plan))
    
    # Calculate savings
    original_savings = user_salary - sum(user_plan.values())
    optimized_savings = user_salary - sum(optimized_plan.values())
    
    print(f"\nOriginal Savings: {original_savings} ({original_savings/user_salary:.1%} of salary)")
    print(f"Optimized Savings: {optimized_savings} ({optimized_savings/user_salary:.1%} of salary)")
    print(f"Improvement: {optimized_savings - original_savings} ({(optimized_savings - original_savings)/original_savings:.1%})")

if __name__ == "__main__":
    example_usage()