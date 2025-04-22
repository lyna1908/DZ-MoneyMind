import math

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
# h(n) = estimated remaining cost to reach goal (optional: use min f(Si) or average deviation)
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
