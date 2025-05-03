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
        self.priorities = priorities
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
        savings_valid = current_savings >= (0.25 * self.salary)  # Aim for 25% savings
        total_valid = abs(total - self.salary) < 0.01
        
        if current_savings > self.best_savings and spending_valid:
            self.best_savings = current_savings
            self.best_node = deepcopy(state)
            
        return spending_valid and savings_valid and total_valid

    def get_valid_actions(self, current_state):
        actions = {}
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
        
        savings_shortfall = max(0, (0.3 * self.salary) - current_savings)
        overspending = max(0, total_spending - (0.9 * self.salary))
        total_deviation = abs(total - self.salary)
        
        priority_weights = {
            'overspending': 1500,
            'savings_shortfall': 800,
            'total_deviation': 1
        }
        
        return (overspending * priority_weights['overspending'] +
                savings_shortfall * priority_weights['savings_shortfall'] +
                total_deviation * priority_weights['total_deviation'])

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
            
            if self.problem.is_goal(node.state):
                self.print_final_results(node)
                return node

            explored.add(hash(node))

            for child in self.problem.expand_node(node, self.use_cost, self.use_heuristic):
                child_hash = hash(child)
                if child_hash not in explored and child_hash not in frontier_states:
                    frontier.put((child.f, child))
                    frontier_states.add(child_hash)
        
        if self.problem.best_node:
            self.print_final_results(BudgetNode(self.problem.best_node), optimal=False)
            return BudgetNode(self.problem.best_node)
        
        print("\nNo valid solution found that meets all constraints")
        return None

    def print_final_results(self, solution, optimal=True):
        initial = self.problem.initial_state
        priorities = self.problem.priorities
        salary = self.problem.salary
        
        print("\n" + "="*60)
        print(" " * 20 + "BUDGET OPTIMIZATION RESULTS")
        print("="*60)
        
        # Print User Priorities
        print("\nUSER PRIORITIES (1 = Highest Priority):")
        print("-"*40)
        for category, priority in sorted(priorities.items(), key=lambda x: x[1]):
            print(f"{category + ':':<15} Priority {priority}")
        
        # Print Initial Plan
        print("\nINITIAL SPENDING PLAN:")
        print("-"*40)
        total_initial = sum(initial.values())
        savings_initial = initial.get('Savings', 0)
        for category in sorted(priorities.keys(), key=lambda x: priorities[x]):
            print(f"{category + ':':<15} {initial[category]:>10.2f}")
        print("-"*40)
        print(f"{'Total:':<15} {total_initial:>10.2f}")
        print(f"{'Spending:':<15} {total_initial-savings_initial:>10.2f}")
        print(f"{'Savings:':<15} {savings_initial:>10.2f} ({savings_initial/salary:.1%})")
        
        # Print Optimal Plan
        opt_label = "OPTIMAL PLAN" if optimal else "BEST FOUND PLAN"
        print(f"\n{opt_label}:")
        print("-"*40)
        total_optimal = sum(solution.state.values())
        savings_optimal = solution.state.get('Savings', 0)
        for category in sorted(priorities.keys(), key=lambda x: priorities[x]):
            change = solution.state[category] - initial[category]
            arrow = "↑" if change > 0 else ("↓" if change < 0 else "→")
            print(f"{category + ':':<15} {solution.state[category]:>10.2f} {arrow} {abs(change):.2f}")
        print("-"*40)
        print(f"{'Total:':<15} {total_optimal:>10.2f}")
        print(f"{'Spending:':<15} {total_optimal-savings_optimal:>10.2f}")
        print(f"{'Savings:':<15} {savings_optimal:>10.2f} ({savings_optimal/salary:.1%})")
        print(f"\nNodes expanded: {self.nodes_expanded}")
        print("="*60 + "\n")

def load_dataset(filename):
    with open(filename, mode='r') as file:
        return list(csv.DictReader(file))

def filter_dataset(dataset, conditions):
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
    
    # User configuration
    user_priorities = {
        'Food': 1,
        'Education': 2,
        'Transport': 3,
        'Savings': 4
    }
    
    initial_plan = {
        'Food': 1200,
        'Education': 400,
        'Transport': 500,
        'Savings': 0
    }
    
    salary = 2000
    
    problem = BudgetProblem(
        initial_state=initial_plan,
        salary=salary,
        filtered_dataset=dataset,
        priorities=user_priorities
    )
    
    print("\nStarting budget optimization...")
    print(f"Target salary: {salary}")
    print(f"Minimum savings target: {0.25*salary} (25%)")
    
    search = GeneralSearch(problem)
    solution = search.search()
    
    if solution:
        print("\nOptimization complete!")
    else:
        print("\nFinished search - no perfect solution found")

if __name__ == "__main__":
    main()