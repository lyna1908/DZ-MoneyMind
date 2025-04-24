import math
import heapq

# --- DEVIATION FUNCTION ---
# f(Si) = | Si / T - ( a * (max(P) - Pi + 1) ) |
def deviation(Si, Pi, maxP, salary, a):
    """Returns the absolute deviation from ideal spending based on priority."""
    ideal = a * (maxP - Pi + 1)
    return abs((Si / salary) - ideal)

# --- SAVINGS PRIORITY FUNCTION ---
# P_savings = (salary / 25000) - 0.75 * number_of_children
def savings_priority(salary, num_children):
    """Returns a priority score for savings based on income and children."""
    return (salary / 25000) - 0.75 * num_children

# --- NORMALIZATION FACTOR a ---
# a = 1 / [ (max(P) - P_savings + 1) + ∑(max(P) - Pi + 1) ]
def compute_a(priorities: dict, P_savings, maxP):
    """Computes the normalization factor 'a' so that ideal percentages sum to 1."""
    sum_terms = sum((maxP - Pi + 1) for Pi in priorities.values())
    return 1 / ((maxP - P_savings + 1) + sum_terms)

# --- CATEGORY COST FUNCTION ---
# cost(Si, Pi) = Si * f(Si) * Pi
def cost(Si, Pi, salary, a, maxP):
    """Returns the cost contribution for a category."""
    return Si * deviation(Si, Pi, maxP, salary, a) * Pi

# --- TOTAL COST FUNCTION ---
# Total = ∑(Si * f(Si) * Pi) + (T - ∑Si) * P_savings
def total_cost(spending: dict, priorities: dict, salary, P_savings):
    """Computes the total cost of a full spending plan."""
    maxP = max(priorities.values())
    a = compute_a(priorities, P_savings, maxP)
    total_spent = sum(spending.values())
    sum_costs = sum(
        cost(Si, priorities[cat], salary, a, maxP)
        for cat, Si in spending.items()
    )
    return sum_costs + (salary - total_spent) * P_savings

# --- PATH COST FUNCTION ---
# g(n): sum of cost() for all categories up to node n
def path_cost(path: list, priorities: dict, salary, P_savings):
    """Returns cumulative path cost for a given path (list of spending dicts)."""
    return sum(total_cost(node, priorities, salary, P_savings) for node in path)

# --- HEURISTIC FUNCTION ---
# h(n) = estimated remaining cost to reach goal
def heuristic(current_node_spending: dict, priorities: dict, salary, P_savings):
    """Estimate remaining cost from a partial node (can be refined)."""
    maxP = max(priorities.values())
    a = compute_a(priorities, P_savings, maxP)
    return sum(
        deviation(Si, priorities[cat], maxP, salary, a) * Pi
        for cat, Si in current_node_spending.items()
        for Pi in [priorities[cat]]
    )

# --- TOTAL A* COST FUNCTION ---
# f(n) = g(n) + h(n)
def a_star_score(path: list, current_node: dict, priorities: dict, salary, P_savings):
    """Returns A* total cost score: path cost + heuristic estimate."""
    return path_cost(path, priorities, salary, P_savings) + heuristic(current_node, priorities, salary, P_savings)

# --- SCORE FUNCTION (End Plan Evaluation) ---
# Pure scoring formula for final plan comparison
def score_function(spending: dict, priorities: dict, salary, P_savings):
    """Final score of a full plan. Can be used for ranking or goal test."""
    return total_cost(spending, priorities, salary, P_savings)

# --- BRANCH FACTOR FUNCTION ---
# Example: b = floor(sqrt(salary / 10000))
def branch_factor(salary):
    """Determines how many children to generate per node (affects tree width)."""
    return max(2, int(math.sqrt(salary / 10000)))

# --- NODE CLASS FOR A* SEARCH ---
class Node:
    def __init__(self, spending, path=None):
        self.spending = spending  # Current spending allocation
        self.path = path or []    # Path of spending allocations leading to this node
        self.f_score = 0          # f(n) = g(n) + h(n)
        
    def __lt__(self, other):
        return self.f_score < other.f_score
    
    def __eq__(self, other):
        # Compare allocations for equality
        if not isinstance(other, Node):
            return False
        return self.spending == other.spending
    
    def __hash__(self):
        # Hash based on spending allocation
        return hash(frozenset(self.spending.items()))

# --- GENERATE NEIGHBORING STATES ---
def generate_neighbors(current_node, priorities, salary, P_savings, max_adjustment=0.05):
    """
    Generate neighboring spending allocations by adjusting each category.
    Adjustments are constrained to ensure allocations remain valid.
    """
    neighbors = []
    bf = branch_factor(salary)
    spending = current_node.spending
    categories = list(spending.keys())
    
    # Calculate total spent and available to spend
    total_spent = sum(spending.values())
    available = salary - total_spent
    
    # For each category, try different spending adjustments
    for category in categories:
        current_amount = spending[category]
        
        # Calculate adjustment step size based on branch factor
        step_size = (max_adjustment * salary) / bf
        
        # Generate bf different adjustments for this category
        for i in range(1, bf + 1):
            # Try increasing spending in this category
            if available >= step_size * i:
                new_spending = spending.copy()
                new_spending[category] = current_amount + step_size * i
                neighbors.append(Node(new_spending, current_node.path + [spending]))
            
            # Try decreasing spending in this category
            if current_amount >= step_size * i:
                new_spending = spending.copy()
                new_spending[category] = current_amount - step_size * i
                neighbors.append(Node(new_spending, current_node.path + [spending]))
    
    # Consider adding to savings (not spending)
    if available > 0:
        new_spending = spending.copy()
        # We don't modify any category - the difference remains as savings
        neighbors.append(Node(new_spending, current_node.path + [spending]))
    
    return neighbors

# --- GOAL TEST ---
def is_goal(node, priorities, salary, P_savings, threshold=0.01):
    """
    Test if a node represents a goal state.
    A goal state has minimal deviation from ideal allocation.
    """
    maxP = max(priorities.values())
    a = compute_a(priorities, P_savings, maxP)
    
    # Calculate average deviation from ideal allocation
    total_deviation = sum(
        deviation(Si, priorities[cat], maxP, salary, a)
        for cat, Si in node.spending.items()
    )
    avg_deviation = total_deviation / len(node.spending)
    
    # If average deviation is below threshold, consider it a goal
    return avg_deviation < threshold

# --- A* SEARCH ALGORITHM ---
def a_star_search(initial_spending, priorities, salary, num_children, max_iterations=1000):
    """
    A* search to find optimal spending allocation.
    
    Args:
        initial_spending: Dictionary of initial category allocations
        priorities: Dictionary of priority values for each category
        salary: Total monthly income
        num_children: Number of children (affects savings priority)
        max_iterations: Maximum number of iterations to prevent infinite loops
    
    Returns:
        Best spending allocation found or None if no solution
    """
    # Calculate savings priority
    P_savings = savings_priority(salary, num_children)
    
    # Create initial node
    initial_node = Node(initial_spending)
    initial_node.f_score = a_star_score(
        [initial_spending], initial_spending, priorities, salary, P_savings
    )
    
    # Open and closed sets
    open_set = [initial_node]
    heapq.heapify(open_set)
    closed_set = set()
    
    iterations = 0
    best_node = initial_node
    best_score = initial_node.f_score
    
    while open_set and iterations < max_iterations:
        current_node = heapq.heappop(open_set)
        
        # Update best node if this one has a better score
        current_score = score_function(current_node.spending, priorities, salary, P_savings)
        if current_score < best_score:
            best_node = current_node
            best_score = current_score
        
        # Check if goal reached
        if is_goal(current_node, priorities, salary, P_savings):
            return current_node.spending
        
        # Add to closed set
        closed_set.add(current_node)
        
        # Generate neighbors
        neighbors = generate_neighbors(current_node, priorities, salary, P_savings)
        
        for neighbor in neighbors:
            # Skip if already in closed set
            if neighbor in closed_set:
                continue
            
            # Calculate f-score for neighbor
            neighbor.f_score = a_star_score(
                neighbor.path, neighbor.spending, priorities, salary, P_savings
            )
            
            # Add to open set
            existing = False
            for i, node in enumerate(open_set):
                if node == neighbor and node.f_score > neighbor.f_score:
                    # Replace existing node with better path
                    open_set[i] = neighbor
                    heapq.heapify(open_set)
                    existing = True
                    break
            
            if not existing:
                heapq.heappush(open_set, neighbor)
        
        iterations += 1
        
    # Return best allocation found
    print(f"A* terminated after {iterations} iterations")
    return best_node.spending

# --- TESTING FUNCTION ---
def test_a_star_budget_allocation():
    """
    Test the A* budget allocation algorithm with sample data.
    """
    # Sample data
    salary = 5000  # Monthly income
    num_children = 1
    
    # Priorities for different spending categories (higher = more important)
    priorities = {
        "Housing": 5,
        "Food": 4,
        "Transportation": 3,
        "Entertainment": 2,
        "Utilities": 4
    }
    
    # Initial allocation (start with equal distribution)
    initial_spending = {
        "Housing": 1500,
        "Food": 800,
        "Transportation": 400,
        "Entertainment": 300,
        "Utilities": 500
    }
    
    print("Initial spending allocation:")
    for category, amount in initial_spending.items():
        print(f"{category}: ${amount} ({amount/salary*100:.1f}%)")
    
    # Calculate initial metrics
    P_savings = savings_priority(salary, num_children)
    initial_cost = total_cost(initial_spending, priorities, salary, P_savings)
    print(f"Initial cost: {initial_cost:.2f}")
    print(f"Savings priority: {P_savings:.2f}")
    
    # Run A* search
    print("\nRunning A* search for optimal allocation...")
    optimal_spending = a_star_search(initial_spending, priorities, salary, num_children)
    
    if optimal_spending:
        print("\nOptimal spending allocation:")
        total_allocated = sum(optimal_spending.values())
        savings = salary - total_allocated
        
        for category, amount in optimal_spending.items():
            print(f"{category}: ${amount:.2f} ({amount/salary*100:.1f}%)")
        
        print(f"Savings: ${savings:.2f} ({savings/salary*100:.1f}%)")
        
        # Calculate final cost
        final_cost = total_cost(optimal_spending, priorities, salary, P_savings)
        print(f"Final cost: {final_cost:.2f}")
        print(f"Improvement: {(initial_cost - final_cost) / initial_cost * 100:.1f}%")
    else:
        print("No optimal allocation found.")

# Run the test
if __name__ == "__main__":
    test_a_star_budget_allocation()