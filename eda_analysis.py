import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("data/cleaned_student_data.csv")

print(df.head())

sns.set(style="whitegrid")

plt.figure()
sns.histplot(df["G3"],bins=20,kde=True)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.show()


plt.figure()
sns.scatterplot(x="studytime",y="G3",data=df)
plt.title("Study Time vs Final Grade")
plt.show()

plt.figure()
sns.scatterplot(x="absences",y="G3",data=df)
plt.title("Absences vs Final Grade")
plt.show()

plt.figure()
sns.boxplot(x="subject",y="G3",data=df)
plt.title("Final Grades by Subject")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
