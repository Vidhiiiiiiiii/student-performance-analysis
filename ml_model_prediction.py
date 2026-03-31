import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
warnings.filterwarnings('ignore')

print("="*80)
print("MACHINE LEARNING: STUDENT PERFORMANCE PREDICTION MODEL")
print("="*80)

#step1 load cleaned data

print("\n📂 STEP 1: LOADING CLEANED DATA")
print("="*80)

math_df=pd.read_csv('data/math_cleaned_data.csv')
portuguese_df=pd.read_csv('data/portuguese_cleaned_data.csv')

print(f"✓ Math dataset: {math_df.shape[0]} students, {math_df.shape[1]} features")
print(f"✓ Portuguese dataset: {portuguese_df.shape[0]} students, {portuguese_df.shape[1]} features")

#step2 data prepping

print("\n📊 STEP 2: PREPARING DATA FOR MODELING")
print("="*80)

def prepare_data_for_modeling(df, subject='Math'):
    """
    Prepare data for ML modeling:
    - Handle categorical variables
    - Select features
    - Prepare X (features) and y (target)
    """

    df_copy= df.copy()

    #target var
    y= df_copy['G3']

    #define feature to use(exclude identifiers and past grades that are too predictive)
    #we exclude G1 nd G2 because theyre almost perfect predictors
    #and would make the model less interesting for research

    categorical_cols=df_copy.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_cols= df_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()

    #remove target and identifier columns
    numeric_cols= [col for col in numeric_cols if col not in ['G3', 'G1', 'G2']]

    #encode categorical variables
    X= df_copy[numeric_cols].copy()

    for col in categorical_cols:
        if col in df_copy.columns:
            le= LabelEncoder()
            X[col]= le.fit_transform(df_copy[col].astype(str))

    #handle missing values
    X= X.fillna(X.mean())

    print(f"✓ {subject}: Selected {X.shape[1]} features")
    print(f"  Features: {', '.join(X.columns[:5])}...")
    print(f"  Target shape: {y.shape}")

    return X, y, X.columns

#prep data for both subs
X_math, y_math, feature_names_math= prepare_data_for_modeling(math_df, 'Math')
X_por, y_por, feature_names_por= prepare_data_for_modeling(portuguese_df, 'Portuguese')

#step3 split data into train/test

print("\n🔀 STEP 3: SPLITTING DATA (80% Train, 20% Test)")
print("="*80)

X_math_train, X_math_test, y_math_train, y_math_test= train_test_split(
    X_math, y_math, test_size=0.2, random_state=42
)

X_por_train, X_por_test, y_por_train, y_por_test= train_test_split(
    X_por, y_por, test_size=0.2, random_state=42
)

print(f"✓ Math: {len(X_math_train)} train, {len(X_math_test)} test")
print(f"✓ Portuguese: {len(X_por_train)} train, {len(X_por_test)} test")

#step4 build models

print("\n🤖 STEP 4: BUILDING PREDICTION MODELS")
print("="*80)

def build_and_evaluate_models(X_train, X_test, y_train, y_test, subject='Math', feature_names=None):
    """
    Build multiple models and evaluate performance
    """

    results= {

    }

    #standardize features for linear regression
    scaler= StandardScaler()
    X_train_scaled= scaler.fit_transform(X_train)
    X_test_scaled= scaler.transform(X_test)

    #model1 linear regression

    print(f"\n 📈 Building Linear Regression ({subject})...")
    lr_model= LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    y_pred_lr= lr_model.predict(X_test_scaled)

    lr_r2= r2_score(y_test, y_pred_lr)
    lr_rmse= np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_mae= mean_absolute_error(y_test, y_pred_lr)

    print(f"    ✓ R² Score: {lr_r2:.4f}")
    print(f"    ✓ RMSE: {lr_rmse:.4f}")
    print(f"    ✓ MAE: {lr_mae:.4f}")

    results['Linear Regression']= {
        'model': lr_model,
        'r2': lr_r2,
        'rmse': lr_rmse,
        'mae': lr_mae,
        'predictions': y_pred_lr,
        'coefficients': lr_model.coef_,
        'feature_names': feature_names
    }

    #model2 random forest

    print(f"\n  🌲 Building Random Forest ({subject})...")
    rf_model= RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    y_pred_rf= rf_model.predict(X_test)

    rf_r2= r2_score(y_test, y_pred_rf)
    rf_rmse= np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_mae= mean_absolute_error(y_test, y_pred_rf)

    print(f"    ✓ R² Score: {rf_r2:.4f}")
    print(f"    ✓ RMSE: {rf_rmse:.4f}")
    print(f"    ✓ MAE: {rf_mae:.4f}")

    results['Random Forest']= {
        'model': rf_model,
        'r2': rf_r2,
        'rmse': rf_rmse,
        'mae': rf_mae,
        'predictions': y_pred_rf,
        'importances': rf_model.feature_importances_,
        'feature_names': feature_names
    }

    return results

#build models for both subjects

print("\n"+"="*80)
print("BUILDING MODELS FOR MATH")
print("="*80)
math_models= build_and_evaluate_models(X_math_train, X_math_test, y_math_train, y_math_test, 'Math', feature_names_math )
print("\n"+"="*80)
print("BUILDING MODELS FOR PORTUGUESE")
print("="*80)
portuguese_models= build_and_evaluate_models(X_por_train, X_por_test, y_por_train, y_por_test, 'Portuguese', feature_names_por)

#step5 model comparison

print("\n"+"="*80)
print("📊 MODEL PERFORMANCE COMPARISON")

print("="*80)

comparison_data= []

for model_name in math_models.keys():
    comparison_data.append({
        'Model': model_name,
        'Subject': 'Math',
        'R² Score': f"{math_models[model_name]['r2']:.4f}",
        'RMSE': f"{math_models[model_name]['rmse']:.4f}",
        'MAE': f"{math_models[model_name]['mae']:.4f}"
    })

for model_name in portuguese_models.keys():
    comparison_data.append({
        'Model': model_name,
        'Subject': 'Portuguese',
        'R² Score': f"{portuguese_models[model_name]['r2']:.4f}",
        'RMSE': f"{portuguese_models[model_name]['rmse']:.4f}",
        'MAE': f"{portuguese_models[model_name]['mae']:.4f}"
    })

comparison_df= pd.DataFrame(comparison_data)
print("\n"+ comparison_df.to_string(index=False))

#step6 feature importance analysis

print("\n"+"="*80)
print("🎯 FEATURE IMPORTANCE ANALYSIS")
print("="*80)

output_dir= 'research_output/ml_analysis'
os.makedirs(output_dir, exist_ok=True)

#feature importance: math

print("\n\n📊 TOP 15 IMPORTANT FEATURES - MATH")
print("="*80)

math_rf= math_models['Random Forest']
importances_math= math_rf['importances']
feature_names_math_list= list(math_rf['feature_names'])

#sort by importance
importance_df_math= pd.DataFrame({
    'Feature': feature_names_math_list,
    'Importance': importances_math
}).sort_values('Importance', ascending=False)

top_15_math= importance_df_math.head(15)
print(top_15_math.to_string(index=False))

#visualize
fig, ax= plt.subplots(figsize=(10,6))
ax.barh(range(len(top_15_math)), top_15_math['Importance'].values, color='steelblue')
ax.set_yticks(range(len(top_15_math)))
ax.set_yticklabels(top_15_math['Feature'].values)
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_title('Top 15 Features Predicting Math Grades\n(Random Forest Model)', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_math.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: feature_importance_math.png")
plt.close()

#feature importance: portuguese
print("\n\n📊 TOP 15 IMPORTANT FEATURES - PORTUGUESE")
print("="*80)

por_rf= portuguese_models['Random Forest']
importances_por= por_rf['importances']
feature_names_por_list= list(por_rf['feature_names'])

importance_df_por= pd.DataFrame({
    'Feature': feature_names_por_list,
    'Importance': importances_por
}).sort_values('Importance', ascending=False)

top_15_por= importance_df_por.head(15)
print(top_15_por.to_string(index=False))

#visualze
fig, ax= plt.subplots(figsize=(10,6))
ax.barh(range(len(top_15_por)), top_15_por['Importance'].values, color='coral')
ax.set_yticks(range(len(top_15_por)))
ax.set_yticklabels(top_15_por['Feature'].values)
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_title('Top 15 Features Predicting Portuguese Grades\n(Random Forest Model)', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_portuguese.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: feature_importance_portuguese.png")
plt.close()

#step7 model predictions visuals

print("\n"+"="*80)
print("📈 PREDICTIONS vs ACTUAL VALUES")
print("="*80)

fig, axes= plt.subplots(1, 2, figsize=(14,5))

#math predictions
ax= axes[0]
ax.scatter(y_math_test, math_models['Random Forest']['predictions'], alpha=0.6, s=30, color='steelblue')
ax.plot([y_math_test.min(), y_math_test.max()], [y_math_test.min(), y_math_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Grade', fontweight='bold')
ax.set_ylabel('Predicted Grade', fontweight='bold')
ax.set_title(f"Math Predictions (R² = {math_models['Random Forest']['r2']:.4f})", fontweight='bold')
ax.grid(True, alpha=0.3)

#portuguese predictions
ax= axes[1]
ax.scatter(y_por_test, portuguese_models['Random Forest']['predictions'], alpha=0.6, s=30, color='coral')
ax.plot([y_por_test.min(), y_por_test.max()], [y_por_test.min(), y_por_test.max()], 'r--', lw=2)
ax.plot([y_por_test.min(), y_por_test.max()], [y_por_test.min(), y_por_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Grade', fontweight='bold')
ax.set_ylabel('Predicted Grade', fontweight='bold')
ax.set_title(f"Portuguese Predictions (R² = {portuguese_models['Random Forest']['r2']:.4f})", fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: predictions_vs_actual.png")
plt.close()

#step8 save model summary report

print("\n"+"="*80)
print("💾 SAVING MODEL SUMMARY REPORT")
print("="*80)

report_file= os.path.join(output_dir, 'ML_Model_Summary.txt')

with open(report_file, 'w') as f:
    f.write("="*80+"\n")
    f.write("MACHINE LEARNING MODEL SUMMARY\n")
    f.write("Student Performance Prediction\n")
    f.write("="*80+"\n\n")

    f.write("DATASET INFORMATION\n")
    f.write("-"*80+"\n")
    f.write(f"Math Students: {len(X_math_train)} train, {len(X_math_test)} test\n")
    f.write(f"Portuguese Students: {len(X_por_train)} train, {len(X_por_test)} test\n")
    f.write(f"Features Used: {X_math.shape[1]}\n\n")
    
    f.write("MODEL PERFORMANCE - MATH\n")
    f.write("-"*90 + "\n")
    for model_name, results in math_models.items():
        f.write(f"\n{model_name}:\n")
        f.write(f"  R² Score: {results['r2']:.4f}\n")
        f.write(f"  RMSE: {results['rmse']:.4f}\n")
        f.write(f"  MAE: {results['mae']:.4f}\n")
    
    f.write("\n\nMODEL PERFORMANCE - PORTUGUESE\n")
    f.write("-"*90 + "\n")
    for model_name, results in portuguese_models.items():
        f.write(f"\n{model_name}:\n")
        f.write(f"  R² Score: {results['r2']:.4f}\n")
        f.write(f"  RMSE: {results['rmse']:.4f}\n")
        f.write(f"  MAE: {results['mae']:.4f}\n")
    
    f.write("\n\nTOP 10 FEATURES - MATH\n")
    f.write("-"*90 + "\n")
    for idx, row in importance_df_math.head(10).iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    
    f.write("\n\nTOP 10 FEATURES - PORTUGUESE\n")
    f.write("-"*90 + "\n")
    for idx, row in importance_df_por.head(10).iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    
    f.write("\n\nKEY INSIGHTS\n")
    f.write("-"*90 + "\n")
    f.write("\n1. MODEL PERFORMANCE:\n")
    f.write(f"   - Math R² Score: {math_models['Random Forest']['r2']:.4f}\n")
    f.write(f"   - Portuguese R² Score: {portuguese_models['Random Forest']['r2']:.4f}\n")
    f.write(f"   - Interpretation: Model explains {max(math_models['Random Forest']['r2'], portuguese_models['Random Forest']['r2'])*100:.1f}% of grade variation\n")
    
    f.write("\n2. MOST IMPORTANT FEATURES:\n")
    f.write(f"   Math: {importance_df_math.iloc[0]['Feature']} ({importance_df_math.iloc[0]['Importance']:.4f})\n")
    f.write(f"   Portuguese: {importance_df_por.iloc[0]['Feature']} ({importance_df_por.iloc[0]['Importance']:.4f})\n")
    
    f.write("\n3. FEATURE DIFFERENCES:\n")
    math_top = set(importance_df_math.head(5)['Feature'].values)
    por_top = set(importance_df_por.head(5)['Feature'].values)
    f.write(f"   - Common Top Features: {math_top.intersection(por_top)}\n")
    f.write(f"   - Math-Specific: {math_top - por_top}\n")
    f.write(f"   - Portuguese-Specific: {por_top - math_top}\n")
 
print(f"✓ Saved: {report_file}")

#step9 summary

print("\n" + "="*90)
print("✅ ML MODEL BUILDING COMPLETE!")
print("="*90)
 
print(f"\nGENERATED FILES:")
print(f"  ✓ Feature importance charts (Math & Portuguese)")
print(f"  ✓ Predictions vs Actual scatter plots")
print(f"  ✓ Model summary report")
print(f"\nOUTPUT DIRECTORY: {output_dir}")
 
print(f"\n\nKEY FINDINGS:")
print(f"  📊 Math Model R²: {math_models['Random Forest']['r2']:.4f}")
print(f"  📊 Portuguese Model R²: {portuguese_models['Random Forest']['r2']:.4f}")
print(f"  🎯 Top Math Feature: {importance_df_math.iloc[0]['Feature']}")
print(f"  🎯 Top Portuguese Feature: {importance_df_por.iloc[0]['Feature']}")
 
print("\n" + "="*90)
print("NEXT STEPS:")
print("="*90)
print("1. Create comparative visualization of feature importance")
print("2. Analyze which factors are unique to each subject")
print("3. Generate final research report with all findings")
print("\n" + "="*90)
