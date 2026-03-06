# import pandas as pd

# math_df=pd.read_csv("data/student-mat.csv",sep=";")
# por_df=pd.read_csv("data/student-por.csv",sep=";")

# print("Math dataset:",math_df.shape)
# print("Portugese dataset:",por_df.shape)

# print("\nColumns:")
# print(math_df.columns)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# load data

print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

math_df=pd.read_csv('data/student-mat.csv',sep=';')
portuguese_df=pd.read_csv('data/student-por.csv',sep=';')

print(f"\n✓ Math dataset loaded: {math_df.shape[0]} students,{math_df.shape[1]} features")
print(f"✓ Portuguese dataset loaded: {portuguese_df.shape[0]} students,{portuguese_df.shape[1]} features")

# remove quote marks

print("\n"+"="*80)
print("STEP 2: CLEANING QUOTE MARKS")
print("="*80)

def clean_quotes(df):
    """Remove quote marks from ll columns"""
    for col in df.columns:
        if df[col].dtype=='object':
            df[col]=df[col].astype(str).str.replace('"','',regex=False)
    return df

math_df=clean_quotes(math_df)
portuguese_df=clean_quotes(portuguese_df)
print("✓ Quote marks removed from all string columns")


# check for missing values

print("\n"+"="*80)
print("STEP 3: CHECKING MISSING VALUES")
print("="*80)

math_missing=math_df.isnull().sum()
portuguese_missing=portuguese_df.isnull().sum()

math_missing_count=math_missing.sum()
portuguese_missing_count=portuguese_missing.sum()

print(f"\nMath dataset - Missing values: {math_missing_count}")
if math_missing_count>0:
    print(math_missing[math_missing>0])
else:
    print("  ✓ No missing values found")

print(f"\nPortuguese dataset - Missing values: {portuguese_missing_count}")
if portuguese_missing_count>0:
    print(portuguese_missing[portuguese_missing>0])
else:
    print("  ✓ No missing values found")

# data type conversion

print("\n"+"="*80)
print("STEP 4: CONVERTING DATA TYPES")
print("="*80)

def convert_data_types(df):
    """Convert columns to appropriate data types"""

    numeric_cols=['age','Medu','Fedu','traveltime','studytime','faliures','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']

    for col in numeric_cols:
        if col in df.columns:
            df[col]=pd.to_numeric(df[col],errors='coerce')

    boolean_cols=['schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']

    for col in boolean_cols:
        for col in df.columns:
            df[col]=df[col].map({'yes':1,'no':0,True:1,False:0})

    categorical_cols=['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian']

    for col in categorical_cols:
        if col in df.columns:
            df[col]=df[col].astype('category')
    
    return df

math_df=convert_data_types(math_df)
portuguese_df=convert_data_types(portuguese_df)
print("✓ Data types converted successfully")

# handel outliers and invalid values

print("\n"+"="*80)
print("STEP 5: HANDLING OUTLIERS AND INVALID VALUES")
print("="*80)

def validate_and_clean(df):
    """Check and handle invalid values"""

    invalid_age=((df['age']<15) | (df['age']>22)).sum()
    if invalid_age>0:
        print(f" ⚠ Found {invalid_age} invalid age values")
        df=df[(df['age']>=15)&(df['age']<=22)]

    grade_cols=['G1','G2','G3']
    for col in grade_cols:
        invalid_grades=((df[col]<0)|(df[col]>20)).sum()
        if invalid_grades >0:
            print(f"  ⚠ Found {invalid_grades} invalid {col} values (outside 0-20)")
            df=df[(df[col]>=0)&(df[col]<=20)]

    invalid_absences=(df['absences']<0).sum()
    if invalid_absences>0:
        print(f" ⚠ Found {invalid_absences} negative absence values")
        df=df[df['absences']>=0]

    return df

math_df=validate_and_clean(math_df)
portuguese_df=validate_and_clean(portuguese_df)
print("✓ Data validated and cleaned")

# feature engineering

print("\n"+"="*80)
print("STEP 6: FEATURE ENGINEERING")
print("="*80)

def engineer_features(df):
    """Create new useful features"""

    df['grade_improvement']=df['G3']-df['G1']

    df['avg_G1_G2']=(df['G1']+df['G2'])/2

    df['weighted_grade']=(df['G1']*0.3+df['G2']*0.7)

    df['grade_volatility']=df[['G1','G2','G3']].std(axis=1)

    df['study_intensity']=(df['studytime']*5)/(df['absences']+1)

    df['parent_education_avg']=(df['Medu']+df['Fedu'])/2

    df['high_achiever']=(df['G3']>=15).astype(int)

    df['struggling']=(df['G3']>=15).astype(int)

    df['support_score']=df['schoolsup']+df['famsup']+df['paid']

    df['alcohol_risk']=((df['Dalc']+df['Walc'])/2>=2).astype(int)

    return df

math_df=engineer_features(math_df)
portuguese_df=engineer_features(portuguese_df)
print("✓ New features created:")
print("  - grade_improvement (G3 - G1)")
print("  - avg_G1_G2 (average of first two periods)")
print("  - weighted_grade (recent performance weighted)")
print("  - grade_volatility (consistency measure)")
print("  - study_intensity (study time vs absences)")
print("  - parent_education_avg (average parental education)")
print("  - high_achiever (G3 >= 15)")
print("  - struggling (G3 < 10)")
print("  - support_score (total support systems)")
print("  - alcohol_risk (high alcohol consumption)")

# summary statistics

print("\n"+"="*80)
print("STEP 7: DATA SUMMARY STATISTICS")
print("="*80)

print("\n📊 MATH DATASET")
print(f"  Shape: {math_df.shape}")
print(f"\n  Grade Statistics (G3 - Final Grade):")
print(f"  Mean: {math_df['G3'].mean():.2f}")
print(f"  Median: {math_df['G3'].median():.2f}")
print(f"  Std Dev: {math_df['G3'].std():.2f}")
print(f"  Min: {math_df['G3'].min():.0f}, Max: {math_df['G3'].max():.0f}")
print(f" High Achievers (G3 >= 15): {math_df['high_achiever'].sum()} ({math_df['high_achiever'].mean()*100:.1f}%)")
print(f" Struggling (G3 < 10): {math_df['struggling'].sum()} ({math_df['struggling'].mean()*100:.1f}%)")
print(f" Avg Absences: {math_df['absences'].mean():.2f}")
print(f" Avg Study Time: {math_df['studytime'].mean():.2f} hours")

print("\n📊 PORTUGUESE DATASET")
print(f"  Shape: {portuguese_df.shape}")
print(f"\n  Grade Statistics (G3 - Final Grade):")
print(f"    Mean: {portuguese_df['G3'].mean():.2f}")
print(f"    Median: {portuguese_df['G3'].median():.2f}")
print(f"    Std Dev: {portuguese_df['G3'].std():.2f}")
print(f"    Min: {portuguese_df['G3'].min():.0f}, Max: {portuguese_df['G3'].max():.0f}")
print(f"  High Achievers (G3 >= 15): {portuguese_df['high_achiever'].sum()} ({portuguese_df['high_achiever'].mean()*100:.1f}%)")
print(f"  Struggling (G3 < 10): {portuguese_df['struggling'].sum()} ({portuguese_df['struggling'].mean()*100:.1f}%)")
print(f"  Avg Absences: {portuguese_df['absences'].mean():.2f}")
print(f"  Avg Study Time: {portuguese_df['studytime'].mean():.2f} hours")

# save cleaned data

print("\n"+"="*80)
print("STEP 8: SAVING CLEANED DATA")
print("="*80)

math_df.to_csv('data/math_cleaned_data.csv',index=False)
portuguese_df.to_csv('data/portuguese_cleaned_data.csv',index=False)

print("✓ Cleaned datasets saved:")
print("data/math_cleaned_data.csv")
print("data/portuguese_cleaned_data.csv")

# display sample data

print("\n"+"="*80)
print("STEP 9: SAMPLE DATA (First 5 rows - Math Dataset)")
print("="*80)
print(math_df.head())

print("\n"+"="*80)
print("DATA CLEANING COMPLESE! ✓")
print("="*80)
print("\nYou can now:")
print("1. Load the cleaned data using:")
print("   math_df = pd.read_csv('data/math_cleaned_data.csv')")
print("   portuguese_df = pd.read_csv('data/portuguese_cleaned_data.csv')")
print("\n2. Proceed to EDA (Exploratory Data Analysis)")
print("3. Build prediction models")