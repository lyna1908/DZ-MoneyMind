import math

class BudgetOptimizer:
    def __init__(self, salary, num_children, priorities):
        """
        Initialize the budget optimizer.
        
        Parameters:
        - salary (float): Total available budget (T)
        - num_children (int): Number of children (affects savings priority)
        - priorities (dict): {category: priority} where lower numbers = higher priority
                            (e.g., {'rent':1, 'food':2})
        """
        self.salary = salary
        self.num_children = num_children
        self.priorities = priorities
        self.max_priority = max(priorities.values()) if priorities else 0
        self.P_savings = self.calculate_savings_priority()
        self.a = self.compute_a()

    def deviation(self, Si, Pi):
        """
        Calculate normalized deviation from ideal spending for a category.
        
        Formula:
        f(Si) = | (Si / T) - [a × (maxP - Pi + 1)] |
        
        Where:
        - Si: Assigned value for category
        - Pi: Priority of category (lower = more important)
        - T: Total salary (self.salary)
        - a: Normalization factor
        - maxP: Maximum priority value in system
        """
        ideal = self.a * (self.max_priority - Pi + 1)
        return abs((Si / self.salary) - ideal)

    def calculate_savings_priority(self):
        """
        Calculate dynamic priority for savings category.
        
        Formula:
        P_savings = (salary / 25000) - (0.75 × num_children)
        
        Rationale:
        - Savings priority increases with higher salary
        - Decreases with more children (more expenses)
        """
        return (self.salary / 25000) - 0.75 * self.num_children

    def compute_a(self):
        """
        Compute normalization factor 'a' ensuring ideal percentages sum to 1.
        
        Formula:
        a = 1 / [ (maxP - P_savings + 1) + Σ(maxP - Pi + 1) ]
        
        Where:
        - Σ sums over all regular categories (non-savings)
        - Ensures ∑(a × (maxP - Pi + 1)) ≈ 1 across all categories
        """
        sum_terms = sum((self.max_priority - Pi + 1) for Pi in self.priorities.values())
        savings_term = (self.max_priority - self.P_savings + 1) if 'savings' not in self.priorities else 0
        return 1 / (savings_term + sum_terms)

    def cost(self, Si, Pi):
        """
        Calculate cost contribution for assigning value Si to category with priority Pi.
        
        Formula:
        cost(Si, Pi) = Si × f(Si) × Pi   +   (T - Si) × (ΣPi + P_savings)/N
        
        Where:
        - Term 1 (Direct cost): Si × deviation × priority
        - Term 2 (Opportunity cost): Remaining budget × average priority
        - N: Number of categories
        - ΣPi: Sum of all priorities
        """
        dev = self.deviation(Si, Pi)
        sum_Pi = sum(self.priorities.values()) + (self.P_savings if 'savings' not in self.priorities else 0)
        remaining = self.salary - Si
        
        term1 = Si * dev * Pi  # Direct cost
        term2 = remaining * (sum_Pi / len(self.priorities)) if self.priorities else 0  # Opportunity cost
        
        return term1 + term2

    def path_cost(self, path):
        """
        Calculate cumulative cost of all assignments in the path (g(n)).
        
        Formula:
        g(n) = Σ cost(Si, Pi) for all assigned categories
        
        Where:
        - path: List of dicts [{'category':..., 'amount':...}, ...]
        - Represents the actual cost of decisions made so far
        """
        return sum(self.cost(node['amount'], self.priorities[node['category']]) for node in path)

    def heuristic(self, current_node):
        """
        Estimate remaining cost to goal using only current node state.
        
        Formula:
        h(n) = Σ [ (remaining_budget × ideal_fraction) × deviation × Pi ]
            for all unassigned categories
        
        Where:
        - ideal_fraction = a × (maxP - Pi + 1)
        - remaining_budget = salary - sum(current_node.values())
        """
        # Calculate remaining budget
        assigned_amount = sum(current_node.values())
        remaining_budget = self.salary - assigned_amount
        
        # Get unassigned categories
        unassigned = [cat for cat in self.priorities if cat not in current_node]
        
        if not unassigned or remaining_budget <= 0:
            return 0
        
        # Calculate heuristic estimate
        estimate = 0
        for category in unassigned:
            Pi = self.priorities[category]
            
            # Calculate ideal fraction for this category
            ideal_frac = self.a * (self.max_priority - Pi + 1)
            
            # Estimate this category's allocation
            estimated_Si = remaining_budget * ideal_frac
            
            # Calculate deviation of estimated allocation
            dev = abs((estimated_Si / self.salary) - ideal_frac)
            
            # Add to total estimate
            estimate += dev * Pi
        
        return estimate

    def a_star_score(self, path, current_node):
        """
        Calculate A* total score: f(n) = g(n) + h(n).
        
        Formula:
        f(n) = path_cost(path) + heuristic(current_node)
        
        Where:
        - g(n): Actual cost of path taken (path_cost)
        - h(n): Estimated remaining cost (heuristic)
        - Used to prioritize nodes in A* search
        """
        return self.path_cost(path) + self.heuristic(current_node)
