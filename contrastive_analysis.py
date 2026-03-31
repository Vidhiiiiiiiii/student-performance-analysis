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

