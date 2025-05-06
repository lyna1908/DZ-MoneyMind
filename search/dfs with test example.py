import random
from collections import defaultdict

# Step 1: Generate a mock dataset
def generate_mock_dataset(num_entries=50):
    dataset = []
    categories = ["food", "bills", "clothing", "transportation", "entertainment"]
    for _ in range(num_entries):
        salary = random.randint(30000, 70000)
        expenses = {
            "food": random.randint(12000, 18000),
            "bills": random.randint(6000, 10000),
            "clothing": random.randint(2000, 7000),
            "transportation": random.randint(4000, 8000),
            "entertainment": random.randint(1000, 6000)
        }
        dataset.append((salary, expenses))
    return dataset

# Step 2: Helper to average expenses from dataset
def average_expenses(filtered_data):
    sums = defaultdict(float)
    counts = defaultdict(int)
    
    for _, expense_dict in filtered_data:
        for cat, val in expense_dict.items():
            sums[cat] += val
            counts[cat] += 1

    return {cat: sums[cat]/counts[cat] for cat in sums}

# Step 3: DFS Budget Optimizer
def dfs_budget_optimizer(user_salary, essentials_min, dataset):
    # Filter relevant salaries
    filtered_data = [entry for entry in dataset if abs(entry[0] - user_salary) <= 2000]
    if not filtered_data:
        print("No matching salaries in dataset.")
        return {}, 0

    base_expenses = average_expenses(filtered_data)

    best_plan = None
    max_savings = -1

    def dfs(current_plan, categories, idx):
        nonlocal best_plan, max_savings
        if idx == len(categories):
            total_spent = sum(current_plan.values())
            if total_spent > user_salary:
                return
            savings = user_salary - total_spent
            if all(current_plan.get(c, 0) >= essentials_min.get(c, 0) for c in essentials_min):
                if savings > max_savings:
                    max_savings = savings
                    best_plan = current_plan.copy()
            return

        category = categories[idx]
        min_val = essentials_min.get(category, 0)
        base_val = base_expenses.get(category, min_val)
        max_val = int(base_val * 1.5)

        for amount in range(int(min_val), max_val + 1, 1000):
            current_plan[category] = amount
            dfs(current_plan, categories, idx + 1)
            del current_plan[category]

    categories = list(base_expenses.keys())
    dfs({}, categories, 0)

    return best_plan, max_savings

# Step 4: Input and run
if __name__ == "__main__":
    user_salary = 50000
    essentials_min = {"food": 15000, "bills": 8000, "transportation": 5000}
    dataset = generate_mock_dataset()

    best_plan, savings = dfs_budget_optimizer(user_salary, essentials_min, dataset)
    
    print("💡 Optimized Spending Plan:")
    for category, amount in best_plan.items():
        print(f"  {category}: {amount} DZD")
    print(f"\n💰 Estimated Monthly Savings: {savings} DZD")
