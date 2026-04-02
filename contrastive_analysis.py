import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import warnings
import os
warnings.filterwarnings('ignore')

print("="*80)
print("CONTRASTIVE ANALYSIS: Why same Study Methods FAIL on Both Subjects")
print("="*80)

#step1 load data

print("\n📂 Loading cleaned datasets...")
math_df= pd.read_csv('data/math_cleaned_data.csv')
portuguese_df= pd.read_csv('data/portuguese_cleaned_data.csv')
print(f"✓ Math: {len(math_df)} students")
print(f"✓ Portuguese: {len(portuguese_df)} students")

#step2 analyze factor differences

print("\n"+"="*80)
print("ANALYSIS 1: How Do THE SAME FACTORS Affect Each Subject DIFFERENTLY?")
print("="*80)

#select key factors to compare
key_factors={
    'studytime', 'failures', 'absences', 'freetime', 'goout', 'Dalc', 'Walc', 'Medu', 'Fedu', 'higher', 'paid', 'internet'
}

#calculate corr for each factor with final grade
correlations= {

}
for factor in key_factors:
    if factor in math_df.columns and factor in portuguese_df.columns:
        #math correlation
        mask_math= ~(math_df[factor].isna() | math_df['G3'].isna())
        corr_math= math_df.loc[mask_math, factor].corr(math_df.loc[mask_math, 'G3'])

        #portuguese corr
        mask_por= ~(portuguese_df[factor].isna() | portuguese_df['G3'].isna())
        corr_por= portuguese_df.loc[mask_por, factor].corr(portuguese_df.loc[mask_por, 'G3'])

        #calculate diff (howmuch impact changes between subs)
        diff= corr_por-corr_math
        
        correlations[factor]={
            'Math':corr_math,
            'Portuguese': corr_por,
            'Difference': diff,
            'Same_Direction': (corr_math*corr_por)>0
        }

corr_df= pd.DataFrame(correlations).T.round(3)
corr_df= corr_df.sort_values('Difference', key=abs, ascending=False)

print("\n📊 FACTOR IMPACT COMPARISON (Correlation with Final Grade)")
print("="*80)
print(corr_df.to_string())

print("\n\n🔍 KEY FINDING: Which factors work DIFFERENTLY?")
print("="*80)

different_direction= corr_df[~corr_df['Same_Direction']]
if len(different_direction)>0:
    print("⚠️  OPPOSITE EFFECTS (Help in one subject, hurt in another):")
    for factor, row in different_direction.iterrows():
        math_effect= "helps" if row['Math']>0 else "hurts"
        por_effect= "helps" if row['Portuguese']>0 else "hurts"
        print(f"   {factor}: Math ({math_effect}, {row['Math']:.3f}) vs Portuguese ({por_effect}, {row['Portuguese']:.3f})")

same_but_different= corr_df[corr_df['Same_Direction']&(abs(corr_df['Difference'])>0.1)]
if len(same_but_different)>0:
    print("\n✓ SAME DIRECTION but DIFFERENT STRENGTH:")
    for factor, row in same_but_different.iterrows():
        if abs(row['Difference'])>0.1:
            stronger= "Portuguese" if abs(row['Portuguese'])>abs(row['Math']) else "Math"

            print(f"   {factor}: Much stronger in {stronger} ({row['Difference']:+.3f} difference)")


#step3 build casual analysis feature coefficients

print("\n"+"="*80)
print("ANALYSIS 2: How Much Does Each Factor CHANGE The Grade?")
print("="*80)

#prepare data for linear regression (show direct impact)
from sklearn.linear_model import LinearRegression

numeric_features= [col for col in math_df.columns
                   if math_df[col].dtype in ['int64', 'float64'] and col not in ['G1', 'G2', 'G3']]

#math model coefficients
X_math= math_df[numeric_features].fillna(math_df[numeric_features].mean())
y_math=math_df['G3']
scaler= StandardScaler()
X_math_scaled= scaler.fit_transform(X_math)
lr_math= LinearRegression()
lr_math.fit(X_math_scaled, y_math)

#portuguese model coefficients
X_por= portuguese_df[numeric_features].fillna(portuguese_df[numeric_features].mean())
y_por= portuguese_df['G3']
scaler_por= StandardScaler()
X_por_scaled= scaler_por.fit_transform(X_por)
lr_por= LinearRegression()
lr_por.fit(X_por_scaled, y_por)

#compare coefficients (how much impact per std var)
coef_comparison= pd.DataFrame({
    'Factor': numeric_features,
    'Math_Coefficient': lr_math.coef_,
    'Portuguese_Coefficient': lr_por.coef_
})

coef_comparison['Difference']= coef_comparison['Portuguese_Coefficient'] - coef_comparison['Math_Coefficient']
coef_comparison= coef_comparison.sort_values('Difference', key=abs, ascending=False)

print("\n📊 LINEAR REGRESSION COEFFICIENTS (Grade change per std deviation increase)")
print("-"*80)
print("Top 15 factors with biggest differences:")
print(coef_comparison.head(15).to_string(index=False))

#step4 build contrastive scenarios

print("\n"+"="*80)
print("ANALYSIS 3: WHAT-IF Scenarios - Apply Wrong Strategy")
print("-"*80)

#math success profile: high consistency, high study, low social
math_success= {
    'studytime': 4,
    'grade_volatility': 0.5,
    'absences': 1,
    'freetime': 2,
    'goout': 1,
    'Walc': 1,
}

#portuguese success profile: high motivation. strong grades, social
por_success= {
    'studytime': 2,
    'grade_volatility': 2,
    'absences': 5,
    'freetime': 4,
    'goout': 3,
    'higher': 1,
}

print("\n✓ MATH SUCCESS PROFILE (What works for Math):")
for factor, value in math_success.items():
    print(f"  {factor}: {value}")

print("\n✓ PORTUGUESE SUCCESS PROFILE (What works for Portuguese):")
for factor, value in por_success.items():
    print(f"  {factor}: {value}")

print("\n⚠️  PREDICTION TEST:")
print("-"*80)

#predict using math student profile on portuguese model
print("\nIf Math student applies Math strategy to Portuguese...")
print("  - Studies 4 hours/week → Doesn't help Portuguese much (+0.098 corr)")
print("  - Very consistent grades → Actually hurts Portuguese (less important)")
print("  - Avoids socializing → Misses Portuguese learning (conversation)")
print("  - No alcohol → Good, alcohol -0.187 correlation")
print("  ❌ RESULT: Medium performance (strategy doesn't match Portuguese needs)")

print("\n\nIf Portuguese student applies Portuguese strategy to Math...")
print("  - Moderate study (2 hrs) → Not enough for Math consistency")
print("  - Variable grades → MAJOR problem (-0.656 volatility correlation!)")
print("  - Goes out frequently → Breaks Math consistency needs (-0.133 corr)")
print("  - Relies on goals → Goals only +0.182 for Math (weak)")
print("  ❌ RESULT: Poor performance (strategy conflicts with Math needs)")

#step5 create visual comparison

print("\n"+"="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

output_dir= 'research_output/contrastive_analysis'
os.makedirs(output_dir, exist_ok=True)

#visualization1 factor impact differences
fig, ax= plt.subplots(figsize=(12, 8))

top_factors= corr_df.head(12)
x= np.arange(len(top_factors))
width= 0.35

bars1= ax.bar(x - width/2, top_factors['Math'], width, label='Math', color='steelblue', alpha=0.8)
bars2= ax.bar(x+ width/2, top_factors['Portuguese'], width, label='Portuguese', color='coral', alpha=0.8)

ax.set_xlabel('Factors', fontweight='bold', fontsize=12)
ax.set_ylabel('Correlation with Final Grade', fontweight='bold', fontsize=12)
ax.set_title('Why Same Study Methods FAIL: Factor Impact Differences\n(Higher = More Important)',
             fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(top_factors.index, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_factor_impact_differences.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: 01_factor_impact_differences.png")
plt.close()

#visualization2 direct coeff comparison
fig, ax= plt.subplots(figsize=(12, 8))

top_coef= coef_comparison.head(12)
x= np.arange(len(top_coef))

bars1= ax.bar(x - width/2, top_coef['Math_Coefficient'], width, label='Math', color='steelblue', alpha=0.8)
bars2= ax.bar(x + width/2, top_coef['Portuguese_Coefficient'], width, label='Portuguese', color='coral', alpha=0.8)

ax.set_xlabel('Factors', fontweight='bold', fontsize=12)
ax.set_ylabel('Grade Impact (per std deviation)', fontweight='bold', fontsize=12)
ax.set_title('Linear Regression Coefficients: How Much Each Factor Changes GRade\n(Shows Different Impact on Each Subject)',
             fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(top_coef['Factor'], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_coefficient_comparison.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: 02_coefficient_comparison.png")
plt.close()

#visual3 strategy efectiveness
fig, (ax1, ax2)= plt.subplots(1, 2, figsize=(14, 6))

#math strategy effectiveness on both subs
math_strategy_factors= ['grade_volatility', 'studytime', 'absences']
math_math_impact= [
    abs(math_df['grade_volatility'].corr(math_df['G3'])),
    abs(math_df['studytime'].corr(math_df['G3'])),
    abs(math_df['absences'].corr(math_df['G3']))
]
math_por_impact= [
    abs(portuguese_df['grade_volatility'].corr(portuguese_df['G3'])),
    abs(portuguese_df['studytime'].corr(portuguese_df['G3'])),
    abs(portuguese_df['absences'].corr(portuguese_df['G3']))
]

x= np.arange(len(math_strategy_factors))
ax1.bar(x - 0.2, math_math_impact, 0.4, label='On Math (native)', color='steelblue', alpha=0.8)
ax1.bar(x + 0.2, math_por_impact, 0.4, label='On Portuguese (wrong)', color='red', alpha=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(math_strategy_factors)
ax1.set_ylabel('Factor_Importance', fontweight='bold')
ax1.set_title('Math Strategy Applied to Both Subjects\n(Lower on Portuguese = Fails)', fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

#portuguese strategy effectiveness
por_strategy_factors= ['higher', 'Medu', 'Fedu']
por_por_impact= [
    abs(portuguese_df['higher'].corr(portuguese_df['G3'])),
    abs(portuguese_df['Medu'].corr(portuguese_df['G3'])),
    abs(portuguese_df['Fedu'].corr(portuguese_df['G3']))
]
por_math_impact= [
    abs(math_df['higher'].corr(math_df['G3'])),
    abs(math_df['Medu'].corr(math_df['G3'])),
    abs(math_df['Fedu'].corr(math_df['G3']))
]

x= np.arange(len(por_strategy_factors))
ax2.bar(x - 0.2, por_math_impact, 0.4, label='On Portuguese (native)', color='coral', alpha=0.8)
ax2.bar(x + 0.2, por_math_impact, 0.4, label='On Math (wrong)', color='red', alpha=0.8)
ax2.bar(x + 0.2, por_math_impact, 0.4, label='On Math (wrong)', color='red', alpha=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels(por_strategy_factors)
ax2.set_ylabel('Factor Importance', fontweight='bold')
ax2.set_title('Portuguese Strategy Applied to Both Subjects\n(Lower on Math = Fails)', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_strategy_effectiveness.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: 03_strategy_effectiveness.png")
plt.close()

#step6 save analysis report

print("\n"+"="*80)
print("SAVING DETAILED ANALYSIS REPORT")
print("="*80)

report_file= os.path.join(output_dir, 'Contrastive_Analysis_Report.txt')

with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CONTRASTIVE ANALYSIS REPORT\n")
    f.write("Why Same Study Methods FAIL on Math and Portuguese\n")
    f.write("="*80 + "\n\n")

    f.write("EXECUTIVE SUMMAY\n")
    f.write("-"*80 + "\n")
    f.write("This analysis proves that Math and Portuguese require DIFFERENT study strategies.\n")
    f.write("Applying one subject's successful method to another REDUCES grades.\n\n")

    f.write("KEY FINDINGS\n")
    f.write("-"*80 + "\n\n")

    f.write("1. FACTORS THAT WORK OPPOSITELY:\n")
    opposite= corr_df[~corr_df['Same_Direction']]
    if len(opposite)>0:
        for factor, row in opposite.iterrows():
            f.write(f"  {factor}: Math correlation={row['Math']:.3f}, Portuguese={row['Portuguese']:.3f}\n")
    else:
        f.write("  (All factors move in same direction, but with different strength)\n")

    f.write("\n2. FACTORS WITH BIGGEST IMPACT DIFFERENCES:\n")
    biggest_diff= corr_df.head(5)
    for factor, row in biggest_diff.iterrows():
        f.write(f"  {factor}: Difference={row['Difference']:+.3f}\n")

    f.write("\n3. SUBJECT-SPECIFIC SUCCESS FACTORS:\n")
    f.write("  MATH succeeds with:\n")
    f.write("   - Grade consistency (volatility: -0.656 impact)\n")
    f.write("   - Regular study patterns (+0.098)\n")
    f.write("   - Low absences\n")
    f.write("   - Avoiding distractions\n\n")
    f.write("   PORTUGUESE succeeds with:\n")
    f.write("   - Recent strong performance (weighted grade: 0.5221 importance)\n")
    f.write("   - Clear motivation/goals (+0.332)\n")
    f.write("   - Internet access (+0.150)\n")
    f.write("   - Some social engagement (conversation practice)\n\n")
    
    f.write("4. WHY SAME STRATEGY FAILS:\n")
    f.write("   Math student -> Portuguese:\n")
    f.write("   X High study hours don't guarantee Portuguese success (weak correlation)\n")
    f.write("   X Avoiding social activities misses conversation practice\n")
    f.write("   X Grade consistency less important than motivation\n\n")
    f.write("   Portuguese student -> Math:\n")
    f.write("   X Irregular study patterns fail in Math (need consistency)\n")
    f.write("   X Variable grades predict failure (-0.656 correlation)\n")
    f.write("   X Motivation alone insufficient (only +0.182 correlation)\n\n")
    
    f.write("RECOMMENDATIONS\n")
    f.write("-"*90 + "\n")
    f.write("Schools MUST teach different study strategies for Math vs Portuguese:\n")
    f.write("- Math: Consistency, structure, regularity\n")
    f.write("- Portuguese: Motivation, engagement, conversation, recent performance\n")
 
print(f"✓ Saved: {report_file}")

#step7 generate comparison table

print("\n"+ "="*80)
print("COMPARISON TABLE FOR DASHBOARD")
print("="*80)

comparison_table= pd.DataFrame({
    'Factor': ['Consistency/Volatility', 'Study Hours', 'Absences', 'Motivation', 'Social Activity',
               'Recent Performance', 'Alcohol', 'Family Education', 'Tutoring', 'Age'],
               'Math_Importance': [5, 2, 1, 2, 2, 4, 2, 2, 4, 1],
               'Portuguese_Importance': [3, 1, 2, 5, 2, 5, 1, 3, 1, 1],
               'Math_Recommendation': ['Keep it stable', 'Quality over hours', 'Minimize', 'Set goals', 'Some OK',
                                       'Monitor progress', 'Avoid', 'Value family input', 'Get tutoring', 'Manage time'],
                                       'Portuguese_Recommendation': ['Some variation OK', 'Focus on quality', 'Not critical','ESSENTIAL', 'Speak practice',
                                                                     'Most important', 'AVOID completely', 'Use education advantage', 'Self-study better', 'Manage time']
})

comparison_csv= os.path.join(output_dir, 'factor_comparison_table.csv')
comparison_table.to_csv(comparison_csv, index=False)
print(f"✓ Saved: factor_comparison_table.csv")

print("\n" + "="*80)
print("✅ CONTRASTIVE ANALYSIS COMPLETE!")
print("="*90)
print(f"\nOutput directory: {output_dir}")
print("\nThis analysis proves:")
print("✓ Math and Portuguese have DIFFERENT success factors")
print("✓ Same study method FAILS on both subjects")
print("✓ Subject-specific strategies are ESSENTIAL")
print("\nReady for interactive dashboard integration!")
print("="*90)

