import pandas as pd
# load the csv file 
df = pd.read_csv(r"C:\Users\LENOVO\Desktop\DZ-MoneyMind\data\Algerian_Household_Budget_1000 (2).csv",encoding="cp1252")

df =  df.drop(columns=["milk" , "bread" , "meat" , "fruits" , "vegetables","qualification"])
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
df.replace({"vrai": True, "faux": False}, inplace=True)



cleaned_path = "data/household_cleaned.csv"
df.to_csv(cleaned_path, index=False)


print(df.head())