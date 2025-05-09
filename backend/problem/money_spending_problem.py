import pandas as pd
from typing import Dict, List, Tuple


class MoneySpendingProblem:
    """
    A class that formulates the money spending management problem as a search problem.
    
    The problem components:
    1. Initial state: An allocation of the user's salary across expense categories, with remainder to savings.
    2. Actions: Adjustments (deltas) to each category (reduce or increase spending).
    3. Transition model: Applying deltas updates the target category and inversely updates savings.
    4. Goal test: Ensures all constraints (bounds, minimum savings, delta limits) are satisfied.
    5. Path cost: Total absolute change from the initial allocation across categories.
    6. Heuristic: Combines savings gap, spending excess, and a delta penalty to guide search.
    """

    def __init__(self, user_salary: float, filtered_data: pd.DataFrame,
                 non_reducible_expenses: List[str] = None):
        """
        Initialize the problem.

        Args:
            user_salary: Monthly salary of the user.
            filtered_data: DataFrame of expense data for similar profiles.
            non_reducible_expenses: Categories that cannot be reduced below a threshold.
        """
        self.user_salary = user_salary
        self.filtered_data = filtered_data

        # Columns to exclude from expense calculations
        exclude_cols = [
            'household_id', 'sector', 'salary', 'own_house', 'own_car',
            'married', 'number_of_children', 'house_rent', 'transport'
        ]
        # Determine the expense categories based on DataFrame columns
        self.expense_categories = [
            col for col in filtered_data.columns if col not in exclude_cols
        ]

        # Default non-reducible expenses if not provided
        self.non_reducible_expenses = non_reducible_expenses or [
            'food', 'healthcare', 'utilities'
        ]

        # Compute mean spending per category from historical data
        self.category_means = self._calculate_category_means()
        # Compute minimum and maximum allowed percentages of salary per category
        self.min_category_percentages = self._calculate_min_percentages()
        self.max_category_percentages = self._calculate_max_percentages()

        # Generate and store the initial allocation state
        self.initial_allocation = self.generate_initial_state()

    def _calculate_category_means(self) -> Dict[str, float]:
        """
        Calculate the mean spending for each category.
        If a category is missing, its mean is treated as 0.0.
        """
        means: Dict[str, float] = {}
        for category in self.expense_categories:
            # Use DataFrame.get to safely retrieve the column or default Series
            means[category] = self.filtered_data.get(
                category, pd.Series([0.0])
            ).mean()
        return means

    def _calculate_min_percentages(self) -> Dict[str, float]:
        """
        Determine the minimum fraction of salary for each category.
        Non-reducible expenses have a higher floor (80% of mean),
        others a lower floor (10% of mean), both normalized by salary.
        """
        min_p: Dict[str, float] = {}
        for category, mean in self.category_means.items():
            # 80% floor for essentials, 10% for others
            factor = 0.8 if category in self.non_reducible_expenses else 0.1
            min_p[category] = factor * (mean / self.user_salary)
        return min_p

    def _calculate_max_percentages(self) -> Dict[str, float]:
        """
        Determine the maximum fraction of salary for each category.
        Savings capped at 50%, others scaled to 120% of historical mean.
        """
        max_p: Dict[str, float] = {}
        for category, mean in self.category_means.items():
            if category == 'savings':
                max_p[category] = 0.5  # Cap savings at 50%
            else:
                max_p[category] = 1.2 * (mean / self.user_salary)
        return max_p

    def generate_initial_state(self) -> Dict[str, float]:
        """
        Allocate the user's salary based on historical means, with remainder to savings.

        Returns:
            A dict mapping each category to its allocated amount.
        """
        initial: Dict[str, float] = {}
        total_allocated = 0.0
        for category in self.expense_categories:
            if category != 'savings':
                amount = self.category_means[category]
                initial[category] = amount
                total_allocated += amount
        # Remainder goes to savings
        initial['savings'] = max(0.0, self.user_salary - total_allocated)
        return initial

    def get_possible_actions(self, state: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Generate all feasible delta-based actions for the current state.

        Each action is (category, delta_amount), where delta_amount is computed as:
        - Reduction: up to 10% of current or to min limit.
        - Increase: up to 10% of current or to max limit.
        """
        actions: List[Tuple[str, float]] = []
        for category in self.expense_categories:
            if category == 'savings':
                continue  # Savings updated indirectly

            current = state[category]
            min_amt = self.min_category_percentages[category] * self.user_salary
            max_amt = self.max_category_percentages[category] * self.user_salary

            # Delta for reducing spending (if allowed)
            if category not in self.non_reducible_expenses and current > min_amt:
                reduction = min(current - min_amt, current * 0.1)
                if reduction > 0:
                    actions.append((category, -reduction))

            # Delta for increasing spending (up to max)
            if current < max_amt:
                increase = min(max_amt - current, current * 0.1)
                if increase > 0:
                    actions.append((category, increase))

        return actions

    def apply_action(self, state: Dict[str, float],
                     action: Tuple[str, float]) -> Dict[str, float]:
        """
        Apply a given delta action to the state.

        Args:
            state: Current allocation mapping.
            action: Tuple of (category, delta_amount).
        Returns:
            New state after applying the delta and adjusting savings inversely.
        """
        category, delta_amount = action
        new_state = state.copy()
        # Update targeted category
        new_state[category] += delta_amount
        # Update savings in opposite direction
        new_state['savings'] -= delta_amount
        return new_state

    def calculate_path_cost(self, state: Dict[str, float]) -> float:
        """
        Compute path cost as sum of absolute changes from the initial allocation.
        Lower cost implies smaller overall adjustments.
        """
        return sum(
            abs(state[cat] - self.initial_allocation[cat])
            for cat in self.expense_categories
        )

    def is_valid_state(self, state: Dict[str, float]) -> bool:
        """
        Check basic validity:
        - Allocations sum to salary (within a small epsilon).
        - No category falls below its minimum required amount.
        """
        if abs(sum(state.values()) - self.user_salary) > 0.01:
            return False
        for category in self.expense_categories:
            if state[category] < self.min_category_percentages[category] * self.user_salary:
                return False
        return True

    def is_goal_state(self, state: Dict[str, float]) -> bool:
        """
        Determine if state meets all goal criteria:
        1. Total equals salary.
        2. Each category within its min-max bounds.
        3. Savings ≥20% of salary.
        4. Non-essential categories within 10% of their minimum.
        5. Total delta ≤30% of salary.
        """
        # 1) Total allocation check
        if abs(sum(state.values()) - self.user_salary) > 0.01:
            return False
        # 2) Bounds for each category
        for category in self.expense_categories:
            amt = state[category]
            if not (
                self.min_category_percentages[category] * self.user_salary
                <= amt <=
                self.max_category_percentages[category] * self.user_salary
            ):
                return False
        # 3) Ensure sufficient savings
        if state.get('savings', 0.0) < 0.2 * self.user_salary:
            return False
        # 4) Non-essential near minimum (within 10%)
        for category in self.expense_categories:
            if category not in self.non_reducible_expenses and category != 'savings':
                min_amt = self.min_category_percentages[category] * self.user_salary
                if state[category] > 1.1 * min_amt:
                    return False
        # 5) Total delta magnitude limit
        total_delta = sum(
            abs(state[c] - self.initial_allocation[c])
            for c in self.expense_categories if c != 'savings'
        )
        if total_delta > 0.3 * self.user_salary:
            return False
        return True

    def heuristic(self, state: Dict[str, float]) -> float:
        """
        Heuristic estimate combining:
        - Savings gap to the 50% cap.
        - Spending excess over minimum for non-essentials.
        - Squared delta penalty to discourage large adjustments.
        """
        # Savings gap: how much more we can save up to 50%
        savings_gap = max(0.0, 0.5 * self.user_salary - state.get('savings', 0.0))

        # Excess spending over minimums
        spending_excess = sum(
            max(0.0, state[cat] - self.min_category_percentages[cat] * self.user_salary)
            for cat in self.expense_categories
            if cat not in self.non_reducible_expenses and cat != 'savings'
        )

        # Delta penalty: squared change per category normalized by salary
        delta_penalty = sum(
            (abs(state[cat] - self.initial_allocation[cat]) ** 2) / self.user_salary
            for cat in self.expense_categories if cat != 'savings'
        )

        return savings_gap + spending_excess + delta_penalty