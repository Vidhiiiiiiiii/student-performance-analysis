import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("STUDENT PERFORMANCE ANALYSIS - PORTUGUESE SUBJECT")
print("="*80)

#step1 load cleaned data

print("\n STEP 1: LOADING CLEANED DATA")
print("-"*80)

portuguese_df=pd.read_csv('data/portuguese_cleaned_data.csv')
print(f"✓ Portuguese dataset loaded: {portuguese_df.shape[0]} students, {portuguese_df.shape[1]} features")


#step2 define variable groups

print("\n📊 STEP 2: DEFINING VARIABLE GROUPS")
print("="*80)

#define groups for research
variable_groups={
    'Academic Behavior':{
        'description':'How students approach learning and dedication to studies',
        'variables':['studytime','failures','absences','G1','G2'],
        'engineered':['study_intensity','grade_improvement','grade_volatility'],
        'questions':[
            'Does more study time directly lead to higher grades?',
            'How much do past failures impact current performance?',
            'What is the relationship between attendance and grades?',
            'Is consistent studying better than sporadic studying?'
        ]
    },

    'Family Background':{
        'description':'Family environment, support, and educational level',
        'variables':['Medu','Fedu','Pstatus','famsize','famsup'],
        'engineered':['parent_education_avg','support_score'],
        'questions':[
            'Does higher parental education increase student performance?',
            'Does family support system matter?',
            'Do larger families perform differently',
            'Is family stability important?'
        ]
    },

    'Socioeconomic & Lifestyle':{
        'description':'Social activities, lesiure, and potential distractions',
        'variables':['address','freetime','goout','romantic','activities','Dalc','Walc'],
        'engineered':['social_activity','alcohol_risk'],
        'questions':[
            'Does more free time hurt academic performance?',
            'How do social activities affect study focus?',
            'Does alcohol consumption impact grades?',
            'Are urban or rural students more focused?'
        ]
    },

    'School Support & Resources':{
        'description':'Academic assistance and tutoring access',
        'variables':['schoolsup','paid','internet','nursery'],
        'engineered':['support_score'],
        'questions':[
            'Does school support improve grades?',
            'Is paid tutoring effective?',
            'Does internet access matter for learning?',
            'Does early education affect performance?'
        ]
    },

    'Motivation & Aspirations':{
        'description':'Student goals and future plans',
        'variables':['higher','reason'],
        'engineered':[],
        'questions':[
            'Do students with higher education goals perform better?',
            'Does reason for course choice affect performance?'
        ]
    },

    'Demographics':{
        'description':'Basic student characteristics',
        'variables':['age','sex','school'],
        'engineered':[],
        'questions':[
            'Does age affect performance?',
            'Is there a gender difference in performance?',
            'Do different schools have different performance patterns?'
        ]
    }
}

print("✓ Variable groups defined:")
for i, (group,info) in enumerate(variable_groups.items(),1):
    print(f"  {i}. {group}")
    print(f"     {info['description']}")


#step3 use existing output directories

output_dir='research_output/'
portuguese_output_dir=os.path.join(output_dir,'portuguese_analysis')
os.makedirs(portuguese_output_dir,exist_ok=True)

print(f"✓ Output directory ready: {portuguese_output_dir}")

#step4 function to generate visualizations for a group

def analyze_group(df, group_name, group_info, subject='Portuguese', save_dir=None):
    """
    Analyze a variable group and generate comprehensive visualizations

    """
    variables= group_info['variables']+group_info['engineered']
    questions= group_info['questions']

    print(f"\n{'='*80}")
    print(f"📊 ANALYZING: {group_name.upper()} ({subject})")
    print(f"{'='*80}")

    #visual1 distribution of variables
    valid_vars= [v for v in variables if v in df.columns]

    if len(valid_vars)>0:
        fig, axes= plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{subject} - {group_name}: Distribution of Variables',
                     fontsize=16, fontweight='bold', y=1.00)
        
        axes= axes.flatten()

        for idx, var in enumerate(valid_vars[:4]):
            ax= axes[idx]

            if df[var].dtype in ['float64', 'int64']:
                ax.hist(df[var], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
                ax.set_xlabel(var, fontsize=11, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(f'{var}\n(Mean: {df[var].mean():.2f}, Std: {df[var].std():.2f})',
                             fontsize=10)
                ax.grid(axis='y', alpha=0.3)
            else:
                value_counts=df[var].value_counts()
                ax.bar(range(len(value_counts)), value_counts.values, color='coral', edgecolor='black', alpha=0.7)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45)
                ax.set_xlabel(var, fontsize=11, fontweight='bold')
                ax.set_ylabel('Count', fontsize=11)
                ax.set_title(f'{var}\n(Categories: {len(value_counts)})', fontsize=10)
                ax.grid(axis='y', alpha=0.3)

        for idx in range(len(valid_vars),4):
            axes[idx].axis('off')
        
        plt.tight_layout()

        if save_dir:
            filename= f"{group_name.replace(' ', '_').replace('&', 'and')}_01_distributions.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            print(f" ✓ Saved: {filename}")

        plt.close()

    #visual2 corr with final grade (G3)
    numeric_vars= [v for v in valid_vars if df[v].dtype in ['float64', 'int64']]

    if len(numeric_vars)>0:
        fig, axes= plt.subplots(2, 2, figsize=(14,10))
        fig.suptitle(f'{subject} - {group_name}: Impact on Final Grade (G3)',
                     fontsize=16, fontweight='bold', y=1.00)
        
        axes= axes.flatten()

        for idx, var in enumerate(numeric_vars[:4]):
            ax= axes[idx]

            mask= ~(df[var].isna() | df['G3'].isna())
            x= df.loc[mask, var]
            y= df.loc[mask, 'G3']

            if len(x)>0:
                ax.scatter(x, y, alpha=0.5, s=30, color='steelblue', edgecolors='navy')

                z= np.polyfit(x, y, 1)
                p= np.poly1d(z)
                x_trend= np.linspace(x.min(), x.max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')

                corr= x.corr(y)

                ax.set_xlabel(var, fontsize=11, fontweight='bold')
                ax.set_ylabel('Final Grade (G3)', fontsize=11, fontweight='bold')
                ax.set_title(f'{var} vs G3\n(Correlation: {corr:.3f})', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend()

        for idx in range(len(numeric_vars), 4):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_dir:
            filename= f"{group_name.replace(' ', '_').replace('&', 'and')}_02_correlation_G3.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {filename}")

        plt.close()

    #visual3 category comparison
    categorical_vars = [v for v in valid_vars if df[v].dtype in ['object', 'str', 'category']]
    
    if len(categorical_vars) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{subject} - {group_name}: Performance by Category', 
                     fontsize=16, fontweight='bold', y=1.00)
        
        axes = axes.flatten()
        
        for idx, var in enumerate(categorical_vars[:4]):
            ax = axes[idx]
            
            df_temp = df[[var, 'G3']].dropna()
            if len(df_temp) > 0:
                df_temp.boxplot(column='G3', by=var, ax=ax)
                ax.set_xlabel(var, fontsize=11, fontweight='bold')
                ax.set_ylabel('Final Grade (G3)', fontsize=11, fontweight='bold')
                ax.set_title(f'Grades by {var}', fontsize=10)
                plt.sca(ax)
                plt.xticks(rotation=45)
        
        for idx in range(len(categorical_vars), 4):
            axes[idx].axis('off')
        
        plt.suptitle('')
        fig.suptitle(f'{subject} - {group_name}: Performance by Category', 
                    fontsize=16, fontweight='bold', y=1.00)
        
        plt.tight_layout()
        
        if save_dir:
            filename = f"{group_name.replace(' ', '_').replace('&', 'and')}_03_category_comparison.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {filename}")
        
        plt.close()

    #summary stats
    print(f"\n  📈 Summary Statistics for {group_name}:")
    print(f" {'-'*75}")

    summary_stats= []

    for var in numeric_vars:
        if var in df.columns:
            mask= ~df[var].isna()
            if mask.sum()>0:
                stats_dict= {
                    'Variable': var,
                    'Mean': f"{df.loc[mask, var].mean():.2f}",
                    'Std Dev': f"{df.loc[mask, var].std():.2f}",
                    'Min': f"{df.loc[mask, var].min():.2f}",
                    'Max': f"{df.loc[mask, var].max():.2f}",
                    'Corr with G3': f"{df.loc[mask, var].corr(df.loc[mask, 'G3']):.3f}"

                }
                summary_stats.append(stats_dict)
                print(f" {var}: Mean={stats_dict['Mean']}, Corr with G3={stats_dict['Corr with G3']}")
    print()

    return summary_stats


#step5 Analyze each group for portuguese

print("\n" + "="*80)
print("🎯 STARTING PORTUGUESE ANALYSIS")
print("="*80)

all_summaries_portuguese={}

for group_name, group_info in variable_groups.items():
    try:
        summary= analyze_group(portuguese_df, group_name, group_info,
                               subject='Portuguese', save_dir=portuguese_output_dir)
        all_summaries_portuguese[group_name]= summary
    except Exception as e:
        print(f"⚠ Error analyzing {group_name}: {str(e)}")
        continue

#step6 save research summary for portuguese

print("\n"+"="*80)
print("💾 SAVING RESEARCH SUMMARY")
print("="*80)

summary_file= os.path.join(portuguese_output_dir, 'resarch_summary_portuguese.txt')

with open(summary_file, 'w') as f:
    f.write("="*80+"\n")
    f.write("STUDENT PERFORMANCE ANALYSIS - PORTUGUESE SUBJECT\n")
    f.write("RESEARCH SUMMARY\n")
    f.write("="*90 + "\n\n")
    
    for group_name, group_info in variable_groups.items():
        f.write(f"\n{group_name.upper()}\n")
        f.write(f"{'-'*90}\n")
        f.write(f"Description: {group_info['description']}\n\n")
        f.write(f"Research Questions:\n")
        for i, q in enumerate(group_info['questions'], 1):
            f.write(f"  {i}. {q}\n")
        f.write(f"\nVariables Analyzed: {', '.join(group_info['variables'] + group_info['engineered'])}\n")
        f.write("\n")
 
print(f"✓ Research summary saved to: {summary_file}")

#step7 overall statistics

print("\n" + "="*90)
print("📊 PORTUGUESE DATASET - OVERALL STATISTICS")
print("="*90)
 
print(f"\nDataset Shape: {portuguese_df.shape}")
print(f"Total Students: {len(portuguese_df)}")
print(f"\nFinal Grade (G3) Statistics:")
print(f"  Mean: {portuguese_df['G3'].mean():.2f}")
print(f"  Median: {portuguese_df['G3'].median():.2f}")
print(f"  Std Dev: {portuguese_df['G3'].std():.2f}")
print(f"  Min: {portuguese_df['G3'].min():.0f}")
print(f"  Max: {portuguese_df['G3'].max():.0f}")
 
print(f"\nStudent Categories:")
print(f"  High Achievers (G3 >= 15): {(portuguese_df['G3'] >= 15).sum()} ({(portuguese_df['G3'] >= 15).mean()*100:.1f}%)")
print(f"  Average (10 <= G3 < 15): {((portuguese_df['G3'] >= 10) & (portuguese_df['G3'] < 15)).sum()} ({((portuguese_df['G3'] >= 10) & (portuguese_df['G3'] < 15)).mean()*100:.1f}%)")
print(f"  Struggling (G3 < 10): {(portuguese_df['G3'] < 10).sum()} ({(portuguese_df['G3'] < 10).mean()*100:.1f}%)")
 
print("\n" + "="*90)
print("✅ PORTUGUESE ANALYSIS COMPLETE!")
print("="*90)
print(f"\nAll visualizations saved to: {portuguese_output_dir}")
print(f"\nNext Step: Compare Math vs Portuguese patterns")
print(f"Then: Build ML models for both subjects")
print("\n" + "="*90)