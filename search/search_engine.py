import queue
from copy import deepcopy
import csv

class BudgetNode:
    def __init__(self, state, parent=None, action=None, g=0, f=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.f = f
        self.depth = 0 if parent is None else parent.depth + 1

    def __hash__(self):
        return hash(frozenset(self.state.items()))

    def __eq__(self, other):
        return isinstance(other, BudgetNode) and self.state == other.state

    def __gt__(self, other):
        return self.f > other.f

class BudgetProblem:
    def __init__(self, initial_state, salary, filtered_dataset, priorities):
        self.initial_state = initial_state
        self.salary = salary
        self.dataset = filtered_dataset
        self.priorities = priorities  # Dictionary of {category: priority_weight}
        self.dataset_avg = self._calculate_dataset_averages()
        self.best_savings = 0
        self.best_node = None

    def _calculate_dataset_averages(self):
        if not self.dataset:
            return {}
            
        categories = [c for c in self.dataset[0].keys() 
                     if c not in {'household_id', 'sector', 'salary', 'own_house', 
                                 'house_rent', 'own_car', 'car_expenses', 'married', 
                                 'number_of_children'}]
        
        averages = {}
        for category in categories:
            values = [float(row[category]) for row in self.dataset if row[category]]
            averages[category] = sum(values) / len(values) if values else 0
            
        return averages

    def is_goal(self, state):
        total_spending = sum(v for k, v in state.items() if k != 'Savings')
        current_savings = state.get('Savings', 0)
        total = total_spending + current_savings
        
        spending_valid = total_spending <= self.salary
        savings_valid = current_savings >= (0.2 * self.salary)
        total_valid = abs(total - self.salary) < 0.01
        
        if current_savings > self.best_savings and spending_valid:
            self.best_savings = current_savings
            self.best_node = state
            
        return spending_valid and savings_valid and total_valid

    def get_valid_actions(self, current_state):
        actions = {}
        # Sort categories by priority (highest priority first)
        sorted_categories = sorted(self.priorities.keys(),
                                 key=lambda x: self.priorities[x])
        
        for category in sorted_categories:
            if category in current_state:
                unique_values = {float(row[category]) for row in self.dataset 
                               if row[category] and float(row[category]) != current_state[category]}
                actions[category] = sorted(unique_values)
            
        return actions

    def expand_node(self, node, use_cost=True, use_heuristic=False):
        children = []
        current_state = node.state
        
        for category, values in self.get_valid_actions(current_state).items():
            for value in values:
                new_state = deepcopy(current_state)
                new_state[category] = value
                
                # Calculate cost with priority weighting
                base_cost = abs(value - self.dataset_avg.get(category, 0))
                priority_weight = 1.0 / self.priorities.get(category, 1)
                g = node.g + (base_cost * priority_weight)
                
                h = self.heuristic(new_state) if use_heuristic else 0
                
                children.append(BudgetNode(new_state, node, category, g, g + h))
                
        return children

    def heuristic(self, state):
        total_spending = sum(v for k, v in state.items() if k != 'Savings')
        current_savings = state.get('Savings', 0)
        total = total_spending + current_savings
        
        # Priority-weighted components
        savings_shortfall = max(0, (0.2 * self.salary) - current_savings)
        overspending = max(0, total_spending - self.salary)
        total_deviation = abs(total - self.salary)
        
        # Apply priority weights to violations
        priority_weights = {
            'overspending': 1000,
            'savings_shortfall': 100 * (1 + max(self.priorities.values())),
            'total_deviation': 1
        }
        
        return (overspending * priority_weights['overspending'] +
                savings_shortfall * priority_weights['savings_shortfall'] +
                total_deviation * priority_weights['total_deviation'])

    def print_node(self, node):
        total = sum(node.state.values())
        savings = node.state.get('Savings', 0)
        spending = total - savings
        
        print(f"\nDepth: {node.depth} | Action: {node.action}")
        print("Current Plan:")
        # Print categories in priority order
        for cat in sorted(self.priorities.keys(), key=lambda x: self.priorities[x]):
            if cat in node.state:
                print(f"  {cat} (Priority {self.priorities[cat]}): {node.state[cat]}")
        print(f"Total: {total:.2f} | Spending: {spending:.2f} | Savings: {savings:.2f}")
        print(f"g: {node.g:.2f} | f: {node.f:.2f}")
        print("-" * 60)

class GeneralSearch:
    def __init__(self, problem):
        self.problem = problem
        self.use_cost = True
        self.use_heuristic = True
        self.nodes_expanded = 0

    def search(self, search_strategy="A*"):
        frontier = queue.PriorityQueue()
        explored = set()
        frontier_states = set()

        initial_node = BudgetNode(self.problem.initial_state)
        initial_priority = initial_node.f
        frontier.put((initial_priority, initial_node))
        frontier_states.add(hash(initial_node))

        while not frontier.empty():
            _, node = frontier.get()
            self.nodes_expanded += 1
            
            self.problem.print_node(node)

            if self.problem.is_goal(node.state):
                print(f"\n>>> GOAL FOUND AFTER {self.nodes_expanded} NODES EXPANDED <<<")
                return node

            explored.add(hash(node))

            for child in self.problem.expand_node(node, self.use_cost, self.use_heuristic):
                child_hash = hash(child)
                if child_hash not in explored and child_hash not in frontier_states:
                    frontier.put((child.f, child))
                    frontier_states.add(child_hash)
        
        if self.problem.best_node:
            print(f"\nBest solution found (savings: {self.problem.best_savings:.2f})")
            return BudgetNode(self.problem.best_node)
        
        print("\nNo valid solution found")
        return None

def load_dataset(filename):
    """Load dataset from CSV file"""
    with open(filename, mode='r') as file:
        return list(csv.DictReader(file))

def filter_dataset(dataset, conditions):
    """Filter dataset based on user conditions"""
    filtered = []
    for row in dataset:
        match = all(str(row[k]) == str(v) for k, v in conditions.items())
        if match:
            filtered.append(row)
    return filtered

def main():
    # Sample test data
    dataset = [
        {'Food': '800', 'Transport': '200', 'Savings': '600', 'Education': '400'},
        {'Food': '900', 'Transport': '350', 'Savings': '500', 'Education': '250'},
        {'Food': '1000', 'Transport': '300', 'Savings': '400', 'Education': '300'}
    ]
    
    for row in dataset:
        for key in row:
            if key in ['Food', 'Transport', 'Savings', 'Education']:
                row[key] = float(row[key])
    
    # User priorities (1 = highest priority)
    user_priorities = {
        'Food': 1,      # Highest priority
        'Education': 2,
        'Transport': 3,
        'Savings': 4    # Lowest priority
    }
    
    initial_plan = {
        'Food': 1200,
        'Education': 400,
        'Transport': 500,
        'Savings': 0
    }
    
    problem = BudgetProblem(
        initial_state=initial_plan,
        salary=2000,
        filtered_dataset=dataset,
        priorities=user_priorities
    )
    
    print("=== STARTING PRIORITY-BASED A* SEARCH ===")
    print("User Priorities:")
    for cat, prio in sorted(user_priorities.items(), key=lambda x: x[1]):
        print(f"  {cat}: Priority {prio}")
    print(f"\nInitial plan totals: {sum(initial_plan.values())}")
    print(f"Target salary: {problem.salary}")
    
    search = GeneralSearch(problem)
    solution = search.search()
    
    if solution:
        print("\n=== OPTIMAL PLAN ===")
        total = sum(solution.state.values())
        savings = solution.state.get('Savings', 0)
        
        # Print in priority order
        for cat in sorted(user_priorities.keys(), key=lambda x: user_priorities[x]):
            print(f"{cat} (Priority {user_priorities[cat]}): {solution.state[cat]:.2f}")
        
        print(f"\nTotal: {total:.2f}")
        print(f"Spending: {total-savings:.2f} (<= {problem.salary:.2f})")
        print(f"Savings: {savings:.2f} (>= {0.2 * problem.salary:.2f})")
    else:
        print("No valid solution found")

if __name__ == "__main__":
    main()