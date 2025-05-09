from collections import deque
from typing import Dict, List, Tuple, Optional

class SearchNode:
    """
    Helper node class for BFS search tree.
    """
    def __init__(self, state: Dict[str, float], parent: Optional['SearchNode'] = None,
                 action: Optional[Tuple[str, float]] = None, depth: int = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth

    def path(self) -> List[Tuple[Dict[str, float], Optional[Tuple[str, float]]]]:
        """
        Reconstruct the path from root to this node.
        Returns a list of (state, action) pairs, where action led to state.
        The first entry has action=None.
        """
        node, p = self, []
        while node is not None:
            p.append((node.state, node.action))
            node = node.parent
        return list(reversed(p))


def breadth_first_search(problem, max_depth: int = 10) -> Optional[List[Tuple[Dict[str, float], Tuple[str, float]]]]:
    """
    Perform BFS on the given MoneySpendingProblem.

    Args:
        problem: An instance of MoneySpendingProblem.
        max_depth: Maximum search depth to avoid infinite loops.

    Returns:
        A list of (state, action) pairs from initial state to goal, or None if no solution found.
    """
    # Initialize frontier with the initial state
    root = SearchNode(problem.initial_allocation)
    if problem.is_goal_state(root.state):
        return root.path()

    frontier = deque([root])
    visited = set()
    # Use tuple of sorted items for hashable representation
    visited.add(tuple(sorted(root.state.items())))

    while frontier:
        node = frontier.popleft()
        if node.depth >= max_depth:
            continue  # Skip nodes deeper than max_depth

        # Expand possible actions
        for action in problem.get_possible_actions(node.state):
            new_state = problem.apply_action(node.state, action)
            state_key = tuple(sorted(new_state.items()))
            # Skip if already seen or invalid
            if state_key in visited or not problem.is_valid_state(new_state):
                continue
            visited.add(state_key)
            child = SearchNode(new_state, parent=node, action=action, depth=node.depth + 1)
            # Check goal
            if problem.is_goal_state(child.state):
                return child.path()
            frontier.append(child)

    # No solution found within depth limit
    return None


# Example usage
if __name__ == '__main__':
    import pandas as pd
    from money_spending_problem import MoneySpendingProblem
    from data.data_filtering import load_data, filter_data
    
    # Load and filter data
    df = load_data()
    filtered_data = filter_data(df, salary=5000.0, own_house=True, own_car=True, number_of_children=0)
    
    if not filtered_data.empty:
        # Create the problem instance
        problem = MoneySpendingProblem(
            user_salary=5000.0,
            filtered_data=filtered_data,
            non_reducible_expenses=['food', 'healthcare', 'utilities']
        )
        
        # Run the search algorithm
        solution = breadth_first_search(problem, max_depth=15)
        
        # Print the results
        if solution:
            print("Solution found:")
            for i, (state, action) in enumerate(solution):
                print(f"Step {i}:")
                if action:  # Skip for initial state
                    print(f"  Action: {action}")
                print(f"  State: {state}")
                print()
        else:
            print("No solution found within depth limit.")
    else:
        print("No matching data found for filtering criteria.")