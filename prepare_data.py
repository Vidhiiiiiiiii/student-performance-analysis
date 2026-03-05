import pandas as pd

math_df=pd.read_csv("data/student-mat.csv",sep=";")
por_df=pd.read_csv("data/student-por.csv",sep=";")

math_df["subject"]="math"
por_df["subject"]="portuguese"

df=pd.concat([math_df,por_df],ignore_index=True)

selected_columns=[
    "studytime","absences","G1","G2","freetime","goout","health","internet","Medu","Fedu","schoolsup","famsup","G3","subject"
]

df=df[selected_columns]

print(df.head())
print("\nDataset shape:",df.shape)

df.to_csv("data/cleaned_student_data.csv",index=False)