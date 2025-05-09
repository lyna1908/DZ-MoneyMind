import pandas as pd

def load_data():
    path = r"C:\Users\LENOVO\Desktop\DZ-MoneyMind\backend\data\household_cleaned.csv"
    df = pd.read_csv(path)

    # Basic cleaning
    df = df.dropna(subset=['salary', 'house_rent', 'transport', 'number_of_children'])

    # Convert types
    df['salary'] = df['salary'].astype(float)
    df['house_rent'] = df['house_rent'].astype(float)
    df['transport'] = df['transport'].astype(float)
    df['number_of_children'] = df['number_of_children'].astype(int)

    # Derive ownership info from rent and transport
    df['own_house'] = df['house_rent'] == 0
    df['own_car'] = df['transport'] == 0

    return df

def filter_data(df, salary=None, own_house=None, own_car=None, number_of_children=None):
    filtered = df.copy()

    if salary is not None:
        min_salary = salary - 2000
        max_salary = salary + 2000
        filtered = filtered[(filtered['salary'] >= min_salary) & (filtered['salary'] <= max_salary)]

    if own_house is not None:
        filtered = filtered[filtered['own_house'] == own_house]
        if own_house:
            filtered = filtered.drop('house_rent', axis=1, errors='ignore')
    else:
        filtered = filtered.drop('house_rent', axis=1, errors='ignore')

    if own_car is not None:
        filtered = filtered[filtered['own_car'] == own_car]
        if own_car:
            filtered = filtered.drop('transport', axis=1, errors='ignore')
        else:
            filtered = filtered.drop('car_expenses', axis=1, errors='ignore')

    if number_of_children is not None:
        filtered = filtered[filtered['number_of_children'] == number_of_children]

    return filtered