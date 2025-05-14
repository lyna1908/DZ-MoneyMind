
import pandas as pd
import queue
import copy
from typing import Dict, List, Tuple
import time

class MoneySpendingProblem:
    def __init__(self, user_salary: float, filtered_data: pd.DataFrame, 
                 non_reducible_expenses: List[str] = None,
                 fixed_values: Dict[str, float] = None):
        self.user_salary = user_salary
        self.filtered_data = filtered_data
        self.non_reducible_expenses = non_reducible_expenses or ['healthcare', 'utilities']
        self.fixed_values = fixed_values or {}

        # Only exclude metadata columns
        exclude_cols = ['household_id', 'sector', 'salary', 'own_house', 'own_car']
        self.expense_categories = [col for col in filtered_data.columns 
                                 if col not in exclude_cols and col != 'savings']

        # Calculate category statistics
        self.category_means = {col: filtered_data[col].mean() 
                              for col in self.expense_categories}
        self.initial_allocation = self.generate_initial_state()

    def generate_initial_state(self) -> Dict[str, float]:
        """Create starting state with fixed values and means"""
        state = {}
        for cat in self.expense_categories:
            state[cat] = self.fixed_values.get(cat, self.category_means[cat])
        state['savings'] = max(0, self.user_salary - sum(state.values()))
        return state

    def get_possible_actions(self, state: Dict[str, float]) -> List[Tuple[str, float]]:
        """Generate reasonable adjustment actions"""
        actions = []
        for cat in self.expense_categories:
            if cat in self.fixed_values:
                continue
                
            current = state[cat]
            mean = self.category_means[cat]
            
            # Possible reductions (10-30% of mean)
            if current > mean * 0.7:
                reduction = min(current - mean*0.7, mean*0.2)
                actions.append((cat, -reduction))
                
            # Possible increases (10-30% of mean)
            if current < mean * 1.3:
                increase = min(mean*1.3 - current, mean*0.2)
                actions.append((cat, increase))
        return actions

    def apply_action(self, state: Dict[str, float], action: Tuple[str, float]) -> Dict[str, float]:
        """Apply an action to create new state"""
        new_state = copy.deepcopy(state)
        category, delta = action
        new_state[category] += delta
        new_state['savings'] -= delta
        return new_state

    def is_valid_state(self, state: Dict[str, float]) -> bool:
        """Check if state meets all constraints"""
        # Verify fixed values
        for cat, val in self.fixed_values.items():
            if abs(state[cat] - val) > 0.01:
                return False
                
        # Check total sum matches salary
        if abs(sum(state.values()) - self.user_salary) > 0.01:
            return False
            
        # Check minimum spending for protected categories
        for cat in self.non_reducible_expenses:
            if state[cat] < self.category_means[cat] * 0.7:
                return False
                
        return True

    def is_goal_state(self, state: Dict[str, float]) -> bool:
        """Check if state meets optimization targets"""
        return (self.is_valid_state(state) and
                state['savings'] >= 0.2 * self.user_salary and
                all(abs(state[cat] - self.category_means[cat]) < self.category_means[cat] * 0.4
                    for cat in self.expense_categories))

    def heuristic(self, state: Dict[str, float]) -> float:
        """Guide search toward optimal allocation"""
        savings_gap = max(0, 0.2*self.user_salary - state['savings'])
        
        spending_penalty = sum(
            max(0, state[cat] - self.category_means[cat]*1.3)
            for cat in self.expense_categories
            if cat not in self.non_reducible_expenses
        )
        
        change_penalty = sum(
            abs(state[cat] - self.category_means[cat]) / self.user_salary
            for cat in self.expense_categories
        )
        
        return savings_gap + spending_penalty + 0.5*change_penalty

class Node:
    def __init__(self, state: Dict[str, float], parent=None, g: float=0, f: float=0):
        self.state = state
        self.parent = parent
        self.g = g  # Path cost
        self.f = f  # Estimated total cost

    def __lt__(self, other):
        return self.f < other.f

class AStarMoneySolver:
    def __init__(self, problem: MoneySpendingProblem):
        self.problem = problem
        self.nodes_explored = 0

    def search(self, max_nodes=10000) -> List[Dict[str, float]]:
        frontier = queue.PriorityQueue()
        explored = set()

        # Ensure we start with non-optimal state if possible
        start_state = self.problem.initial_allocation
        if self.problem.is_goal_state(start_state):
            start_state = self._deoptimize(start_state)
            
        start_node = Node(
            state=copy.deepcopy(start_state),
            g=0,
            f=self.problem.heuristic(start_state)
        )
        frontier.put(start_node)

        while not frontier.empty() and self.nodes_explored < max_nodes:
            current_node = frontier.get()
            self.nodes_explored += 1

            if self.problem.is_goal_state(current_node.state):
                return self._reconstruct_path(current_node)

            state_key = frozenset(current_node.state.items())
            if state_key in explored:
                continue
            explored.add(state_key)

            for action in self.problem.get_possible_actions(current_node.state):
                new_state = self.problem.apply_action(current_node.state, action)
                if not self.problem.is_valid_state(new_state):
                    continue

                new_g = current_node.g + 1
                new_h = self.problem.heuristic(new_state)
                frontier.put(Node(new_state, current_node, new_g, new_g + new_h))

        return None

    def _deoptimize(self, state: Dict[str, float]) -> Dict[str, float]:
        """Create suboptimal starting state if initial is already optimal"""
        new_state = copy.deepcopy(state)
        for cat in self.problem.expense_categories:
            if cat not in self.problem.fixed_values:
                new_state[cat] *= 1.2
        new_state['savings'] = max(0, self.problem.user_salary - sum(
            v for k,v in new_state.items() if k != 'savings'))
        return new_state

    def _reconstruct_path(self, node: Node) -> List[Dict[str, float]]:
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]
    

    







    # Sample dataset
data = [
    {'household_id': 1, 'salary': 60860, 'food': 16164, 'transport': 3219, 'education': 8314, 
     'healthcare': 4872, 'utilities': 4335, 'clothing': 1267, 'entertainment': 4168, 'savings': 18521},
    {'household_id': 2, 'salary': 45311, 'food': 13705, 'transport': 0, 'education': 0, 
     'healthcare': 3171, 'utilities': 2328, 'clothing': 2671, 'entertainment': 1433, 'savings': -16534},
    {'household_id': 3, 'salary': 91551, 'food': 23483, 'transport': 8921, 'education': 0, 
     'healthcare': 8997, 'utilities': 6797, 'clothing': 2946, 'entertainment': 2278, 'savings': 38129},
    {'household_id': 4, 'salary': 108555, 'food': 30522, 'transport': 0, 'education': 11073, 
     'healthcare': 8395, 'utilities': 6029, 'clothing': 6381, 'entertainment': 6378, 'savings': 33625},
    {'household_id': 5, 'salary': 51534, 'food': 14886, 'transport': 0, 'education': 0, 
     'healthcare': 3275, 'utilities': 3857, 'clothing': 1766, 'entertainment': 1754, 'savings': 21980}
]
df = pd.DataFrame(data)

def run_optimization(user_index, fixed_values=None, max_nodes=5000):
    """Run complete optimization test with diagnostics"""
    print(f"\n{'='*40}")
    print(f"Optimizing Budget for User {user_index+1}")
    print(f"{'='*40}")
    
    salary = df.iloc[user_index]['salary']
    print(f"\nAnnual Salary: ${salary:,.2f}")
    print(f"Target Savings (20%): ${0.2*salary:,.2f}")
    
    # Create problem instance
    problem = MoneySpendingProblem(
        user_salary=salary,
        filtered_data=df,
        non_reducible_expenses=['healthcare', 'utilities'],
        fixed_values=fixed_values or {}
    )
    
    # Display initial allocation
    print("\nInitial Allocation:")
    for cat, val in problem.initial_allocation.items():
        print(f"{cat:15}: ${val:,.2f} ({(val/salary*100):.1f}%)")
    
    # Run A* search
    start_time = time.time()
    solver = AStarMoneySolver(problem)
    solution_path = solver.search(max_nodes=max_nodes)
    
    # Analyze results
    if solution_path:
        print("\nOptimization Successful!")
        print(f"Path Length: {len(solution_path)} steps")
        print(f"Nodes Explored: {solver.nodes_explored}")
        
        # Show key steps
        for i in [0, len(solution_path)//2, -1]:
            state = solution_path[i]
            label = "Initial" if i == 0 else "Intermediate" if i < len(solution_path)-1 else "Final"
            print(f"\n{label} State:")
            print(f"Total Savings: ${state['savings']:,.2f} ({(state['savings']/salary*100):.1f}%)")
            print("Major Changes:")
            for cat in problem.expense_categories:
                delta = state[cat] - problem.initial_allocation[cat]
                if abs(delta) > salary*0.01:  # Show changes >1% of salary
                    print(f"  {cat:15}: ${delta:+,.2f} ({(delta/salary*100):+.1f}%)")
    else:
        print("\nOptimization Failed")
        print("Possible Reasons:")
        print("- Initial savings already meets target")
        print("- Fixed values prevent reaching target")
        print("- Need more nodes (increase max_nodes)")
    
    print(f"\nProcessing Time: {time.time()-start_time:.2f}s")

# Run test cases
run_optimization(0)  # User with good initial savings
run_optimization(1, fixed_values={'transport': 2000})  # User with debt
run_optimization(3)  # High-income user