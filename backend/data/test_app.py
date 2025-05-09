import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path to import modules correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from problem.money_spending_problem import MoneySpendingProblem
from problem.search_algo import breadth_first_search

# Import data handling functions - using your existing module
from data_filtering import load_data, filter_data

# Title for the Streamlit app
st.title("🧠 DZ MoneyMind - Smart Money Manager")

# Create sidebar for inputs
with st.sidebar:
    st.header("Your Financial Profile")
    
    # Get user input for the salary, house ownership, car ownership, and number of children
    salary = st.number_input("Monthly Salary (DZD)", min_value=0.0, value=5000.0, step=100.0)
    own_house = st.radio("Do you own a house?", ["Yes", "No"])
    own_car = st.radio("Do you own a car?", ["Yes", "No"])
    number_of_children = st.number_input("Number of children", min_value=0, step=1)
    
    # Multi-select for non-reducible expenses (required categories)
    st.subheader("Non-Reducible Expenses")
    st.caption("These categories will maintain higher minimum thresholds")
    
    # Get the possible expense categories - this will get dynamically updated when data is loaded
    if 'expense_categories' not in st.session_state:
        # Default categories to show before data is loaded
        st.session_state['expense_categories'] = [
            "food", "healthcare", "utilities", "education", "entertainment", "savings"
        ]
        
    non_reducible_expenses = st.multiselect(
        "Select your essential expenses",
        st.session_state['expense_categories'],
        default=["food", "healthcare", "utilities"]
    )

# Main content area
st.header("Money Management Assistant")
st.write("""
This tool helps you optimize your monthly expenses by analyzing similar household patterns
and suggesting adjustments to maximize your savings while meeting essential needs.
""")

# Add tabs for different sections
tab1, tab2, tab3 = st.tabs(["Data Analysis", "Optimization Results", "Recommendations"])

# Load data button
if st.button("Analyze My Financial Profile"):
    with st.spinner("Loading and filtering data..."):
        try:
            # Load the dataset
            df = load_data()
            
            # Apply the filter with the user inputs
            filtered_df = filter_data(
                df,
                salary=salary,
                own_house=(own_house == "Yes"),
                own_car=(own_car == "Yes"),
                number_of_children=number_of_children
            )
            
            # Store the filtered data in session state to access across tabs
            st.session_state['filtered_data'] = filtered_df
            
            # Update expense categories based on actual data columns
            expense_cols = [col for col in filtered_df.columns if col not in 
                          ['household_id', 'sector', 'salary', 'own_house', 'own_car', 
                           'number_of_children', 'house_rent', 'transport']]
            st.session_state['expense_categories'] = expense_cols
            
            # Display filtered data in first tab
            with tab1:
                if filtered_df.empty:
                    st.warning("No similar households found matching your criteria.")
                    st.stop()
                else:
                    st.success(f"Found {len(filtered_df)} similar households")
                    
                    # Show statistics of the filtered data
                    st.subheader("Spending Patterns of Similar Households")
                    
                    # Calculate spending statistics - adjust for your actual column names
                    expense_cols = [col for col in filtered_df.columns if col not in 
                                   ['household_id', 'sector', 'salary', 'own_house', 'own_car', 
                                    'number_of_children', 'house_rent', 'transport']]
                    
                    expense_stats = filtered_df[expense_cols].describe().T[['mean', 'min', 'max']]
                    expense_stats = expense_stats.reset_index()
                    expense_stats.columns = ['Category', 'Average', 'Minimum', 'Maximum']
                    
                    # Format as currency
                    for col in ['Average', 'Minimum', 'Maximum']:
                        expense_stats[col] = expense_stats[col].apply(lambda x: f"{x:.2f} DZD")
                    
                    st.table(expense_stats)
                    
                    # Optional: Show raw data
                    if st.checkbox("Show raw data"):
                        st.dataframe(filtered_df)
            
            # Initialize the problem solver
            with tab2:
                st.subheader("Expense Optimization")
                
                try:
                    # Create MoneySpendingProblem
                    problem = MoneySpendingProblem(
                        user_salary=salary,
                        filtered_data=filtered_df,
                        non_reducible_expenses=non_reducible_expenses
                    )
                    
                    # Initial state display
                    st.write("### Initial Expense Allocation")
                    initial_state = problem.initial_allocation
                    
                    # Create a DataFrame for nicer display
                    initial_df = pd.DataFrame({
                        'Category': initial_state.keys(),
                        'Amount (DZD)': initial_state.values(),
                        'Percentage': [v/salary*100 for v in initial_state.values()]
                    })
                    
                    # Format amounts and percentages
                    initial_df['Amount (DZD)'] = initial_df['Amount (DZD)'].apply(lambda x: f"{x:.2f}")
                    initial_df['Percentage'] = initial_df['Percentage'].apply(lambda x: f"{x:.1f}%")
                    
                    st.table(initial_df)
                    
                    # Run the optimization
                    with st.spinner("Optimizing your expenses..."):
                        max_depth = 15  # Adjust as needed
                        solution = breadth_first_search(problem, max_depth=max_depth)
                    
                    if solution:
                        st.success("Optimization completed successfully!")
                        
                        # Get the final state (optimized allocation)
                        final_state = solution[-1][0]
                        
                        # Create a comparison table
                        comparison = []
                        for category in final_state:
                            initial = initial_state.get(category, 0)
                            final = final_state.get(category, 0)
                            delta = final - initial
                            delta_percent = (delta / initial * 100) if initial > 0 else 0
                            
                            comparison.append({
                                'Category': category,
                                'Initial': f"{initial:.2f}",
                                'Optimized': f"{final:.2f}",
                                'Change': f"{delta:.2f}",
                                'Change %': f"{delta_percent:.1f}%"
                            })
                        
                        # Convert to DataFrame for display
                        comparison_df = pd.DataFrame(comparison)
                        st.write("### Expense Optimization Results")
                        st.table(comparison_df)
                        
                        # Store solution for recommendations tab
                        st.session_state['solution'] = solution
                        st.session_state['final_state'] = final_state
                    else:
                        st.warning("No optimal solution found within the search constraints.")
                        st.info("Try adjusting your non-reducible expense categories or increase the search depth.")
                
                except Exception as e:
                    st.error(f"Error in optimization: {str(e)}")
                    st.exception(e)
            
            # Recommendations tab
            with tab3:
                if 'solution' in st.session_state and 'final_state' in st.session_state:
                    st.subheader("Personalized Financial Recommendations")
                    
                    solution = st.session_state['solution']
                    final_state = st.session_state['final_state']
                    
                    # Display the sequence of actions to take
                    st.write("### Step-by-Step Adjustments")
                    for idx, (state, action) in enumerate(solution[1:], 1):  # Skip initial state
                        if action:
                            category, amount = action
                            direction = "Increase" if amount > 0 else "Decrease"
                            st.write(f"**Step {idx}:** {direction} *{category}* spending by {abs(amount):.2f} DZD")
                    
                    # Show savings impact
                    initial_savings = initial_state.get('savings', 0)
                    final_savings = final_state.get('savings', 0)
                    savings_increase = final_savings - initial_savings
                    savings_percent = (savings_increase / salary) * 100
                    
                    st.write("### Savings Impact")
                    st.metric(
                        label="Monthly Savings",
                        value=f"{final_savings:.2f} DZD",
                        delta=f"{savings_increase:.2f} DZD ({savings_percent:.1f}% of salary)"
                    )
                    
                    # General advice based on the optimization
                    st.write("### Financial Insights")
                    
                    # Find categories with biggest reductions
                    reductions = []
                    for cat in final_state:
                        if cat != 'savings':
                            delta = final_state[cat] - initial_state.get(cat, 0)
                            if delta < 0:
                                reductions.append((cat, delta))
                    
                    # Sort by largest reduction
                    reductions.sort(key=lambda x: x[1])
                    
                    if reductions:
                        st.write("#### Top Areas to Reduce Spending")
                        for cat, delta in reductions[:3]:  # Top 3 reductions
                            st.write(f"- **{cat.title()}**: Reduce by {abs(delta):.2f} DZD")
                    
                    # General financial advice
                    st.write("#### General Advice")
                    st.write("""
                    - Track your expenses regularly to stay within the optimized budget
                    - Consider automating your savings transfer at the beginning of each month
                    - Review your budget quarterly to adapt to changing financial circumstances
                    - Consider investing part of your savings for long-term financial growth
                    """)
                else:
                    st.info("Run the optimization first to see personalized recommendations.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

else:
    # Initial page state
    st.info("Click 'Analyze My Financial Profile' to start the analysis.")