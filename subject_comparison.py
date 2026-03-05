import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("data/cleaned_student_data.csv")

math_df=df[df["subject"]=="math"]
por_df=df[df["subject"]=="portuguese"]

sns.set(style="whitegrid")

# grade distribution comparison

plt.figure(figsize=(8,5))
sns.boxplot(x="subject",y="G3",data=df)
plt.title("Final Grade Distribution: Math vs Portuguese")
plt.savefig("outputs/grade_distribution.png")
plt.show()

# studytime vs grades

fig,axes=plt.subplots(1,2, figsize=(12,5))

sns.scatterplot(ax=axes[0],x="studytime",y="G3",data=math_df)
axes[0].set_title("Math: Studytime vs Final Grade")

sns.scatterplot(ax=axes[1],x="studytime",y="G3",data=por_df)
axes[1].set_title("Portuguese: Studytime vs Final Grade")

plt.tight_layout()
plt.savefig("outputs/studytime_comparison.png")
plt.show()

# absences vs grades

fig,axes=plt.subplots(1,2, figsize=(12,5))

sns.scatterplot(ax=axes[0],x="absences",y="G3",data=math_df)
axes[0].set_title("Math: Absencess vs Grade")

sns.scatterplot(ax=axes[1],x="absences",y="G3",data=por_df)
axes[1].set_title("Portuguese: Absences vs Grade")

plt.tight_layout()
plt.savefig("outputs/absences_comparison.png")
plt.show()

# free time vs grade

fig,axes=plt.subplots(1,2,figsize=(12,5))

sns.boxplot(ax=axes[0],x="freetime",y="G3",data=math_df)
axes[0].set_title("Math: Free Time vs Grade")

sns.boxplot(ax=axes[1],x="freetime",y="G3",data=por_df)
axes[1].set_title("Portuguese: Free Time vs Grade")

plt.tight_layout()
plt.savefig("outputs/freetime_comparison.png")
plt.show()

# goingout vs grade

fig,axes=plt.subplots(1,2,figsize=(12,5))

sns.boxplot(ax=axes[0],x="goout",y="G3",data=math_df)
axes[0].set_title("Math: Going Out vs Grade")

sns.boxplot(ax=axes[1],x="goout",y="G3",data=por_df)
axes[1].set_title("Portuguese: Going Out vs Grade")

plt.tight_layout()
plt.savefig("outputs/goout_comparison.png")
plt.show()

# previous grade vs final grade

fig,axes=plt.subplots(1,2,figsize=(12,5))

sns.scatterplot(ax=axes[0],x="G2",y="G3",data=math_df)
axes[0].set_title("Math: Previous Grade(G2) vs Final Grade")

sns.scatterplot(ax=axes[1],x="G2",y="G3",data=por_df)
axes[1].set_title("Portuguese: Previous Grade(G2) vs Final Grade")

plt.tight_layout()
plt.savefig("outputs/G2_vs_G3.png")
plt.show()

# parent edu vs grade

fig,axes=plt.subplots(1,2,figsize=(12,5))

sns.boxplot(ax=axes[0],x="Medu",y="G3",data=math_df)
axes[0].set_title("Math: Mother Education vs Grade")

sns.boxplot(ax=axes[1],x="Medu",y="G3",data=por_df)
axes[1].set_title("Portuguese: Mother Education vs Grade")

plt.tight_layout()
plt.savefig("outputs/mother_education.png")
plt.show()

# school support vs grade

fig,axes=plt.subplots(1,2,figsize=(12,5))

sns.boxplot(ax=axes[0],x="schoolsup",y="G3",data=math_df)
axes[0].set_title("Math: School Support vs Grade")

sns.boxplot(ax=axes[1],x="schoolsup",y="G3",data=math_df)
axes[0].set_title("Math: School Support vs Grade")

sns.boxplot(ax=axes[1],x="schoolsup",y="G3",data=por_df)
axes[1].set_title("Portuguese: School Support vs Grade")

plt.tight_layout()
plt.savefig("outputs/school_support.png")
plt.show()

# correlation heatmaps

fig,axes=plt.subplots(1,2,figsize=(14,6))

sns.heatmap(math_df.corr(numeric_only=True),ax=axes[0],cmap="coolwarm",annot=True)
axes[0].set_title("Math Correlation Heatmap")

sns.heatmap(por_df.corr(numeric_only=True),ax=axes[1],cmap="coolwarm",annot=True)
axes[1].set_title("Portuguese Correlation Heatmap")

plt.tight_layout()
plt.savefig("outputs/correlation_comparison.png")
plt.show()

# fig,axes=plt.subplots(1,2,figsize=(12,5))

# sns.scatterplot(ax=axes[0],x="studytime",y="G3",data=math_df)
# axes[0].set_title("Math: Studytime vs Final Grade")

# sns.scatterplot(ax=axes[1],x="studytime",y="G3",data=por_df)
# axes[1].set_title("Portuguese: Studytime vs Final Grade")

# plt.tight_layout()
# plt.show()
# plt.savefig("outputs/studytime_comparison.png")


# fig,axes=plt.subplots(1,2,figsize=(12,5))

# sns.scatterplot(ax=axes[0],x="absences",y="G3",data=math_df)
# axes[0].set_title("Math: Absences vs Grades")

# sns.scatterplot(ax=axes[1],x="absences",y="G3",data=por_df)
# axes[1].set_title("Portuguese: Absences vs Grade")

# plt.tight_layout()
# plt.show()
# plt.savefig("outputs/studytime_comparison.png")

# plt.figure(figsize=(8,5))

# sns.boxplot(x="subject",y="G3",data=df)

# plt.title("Final Grade Distribution: Math vs Portuguese")
# plt.show()
# plt.savefig("outputs/studytime_comparison.png")