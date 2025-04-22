import streamlit as st
from data_filtering import load_data, filter_data

# Title for the Streamlit app
st.title("🧠 DZ MoneyMind - Household Filter")

# Get user input for the salary, house ownership, car ownership, and number of children
salary = st.number_input("Enter your salary (DZD)", min_value=0)
own_house = st.radio("Do you own a house?", ["Yes", "No"])
own_car = st.radio("Do you own a car?", ["Yes", "No"])
number_of_children = st.number_input("Enter the number of children", min_value=0, step=1)

# When the filter button is pressed
if st.button("Filter Data"):
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

    # Check if the filtered dataset is empty and display accordingly
    if filtered_df.empty:
        st.warning("No results match your criteria.")
    else:
        st.success(f"Found {len(filtered_df)} matching records:")
        st.dataframe(filtered_df)
