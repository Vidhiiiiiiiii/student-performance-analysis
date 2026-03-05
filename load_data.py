import pandas as pd

math_df=pd.read_csv("data/student-mat.csv",sep=";")
por_df=pd.read_csv("data/student-por.csv",sep=";")

print("Math dataset:",math_df.shape)
print("Portugese dataset:",por_df.shape)

print("\nColumns:")
print(math_df.columns)