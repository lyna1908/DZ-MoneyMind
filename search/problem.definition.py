

class SpendingNode:
    def __init__(self, state, parent=None, level=0, g=0, h=0, dataset_rows=None ):
        self.state = state.copy()
        self.parent = parent
        self.level = level
        self.g = g
        self.h = h
        self.f = g + h
        self.dataset_rows = dataset_rows or []
    def __lt__(self, other):
        return self.f < other.f
    



class SpendingAllocationProblem:
    def __init__(self, salary, priorities, user_input_plan, num_children=0, 
                 has_car=False, has_house=False, fixed_categories=None, dataset_rows=None):
        self.salary = salary
        self.priorities = priorities
        self.categories = list(priorities.keys())
        self.fixed_categories = fixed_categories or {}
        self.user_input_plan = user_input_plan
        self.num_children = num_children
        self.has_car = has_car
        self.has_house = has_house
        self.dataset_rows = dataset_rows or []

        self.initial_state = self.create_initial_state()
        self.max_priority = max(priorities.values())                 
        self.savings_priority = self.calculate_savings_priority()


    def create_initial_state(self):
        """
        Create the initial user plan:
        - Fixed expenses are already set.
        - Other categories are taken from user input or default to 0.
        """
        state = {category: 0 for category in self.categories}
        
        # Fill in fixed values
        for category in self.categories:
            if category in self.fixed_categories:
                state[category] = self.fixed_categories[category]
            else:
                state[category] = 0
        
        return state


    def actions(self, state, level):
        """
        Return all possible actions (i.e., spend values) for the current category.
        """
        if level >= len(self.priorities):
            return []
        
        current_category = self.priorities[level]
        # Extract all unique possible spending values for this category
        return list({row[current_category] for row in self.dataset_rows})

    def transition(self, state, level, action):
        """
        Apply an action to the state: update the spending in the current category.
        """
        new_state = state.copy()
        current_category = self.priorities[level]
        new_state[current_category] = action
        return new_state

    def expand_node(self, parent_node):
        """
        Expand a node by generating all its children using actions and transitions.
        """
        children = []
        level = parent_node.level
        
        for action in self.actions(parent_node.state, level):
            new_state = self.transition(parent_node.state, level, action)
            children.append(SpendingNode(
                state=new_state,
                level=level + 1,
                dataset_rows=self.filtered_rows  # All rows passed down
            ))
        
        return children


    def is_goal_state(state, salary, min_savings=0.1, min_category=1000):
        total = sum(state.values())
        savings = salary - total
        has_underspent = any(v < min_category for v in state.values())
        return (total <= salary and 
             savings >= min_savings * salary and 
             not has_underspent)

    @property
    def sorted_categories(self):
        """Categories sorted by priority (highest first)"""
        return sorted(self.priorities.keys(), key=lambda x: self.priorities[x])