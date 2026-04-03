# app.py
"""
Interactive Dashboard: Student Performance Prediction & Analysis
A comprehensive Streamlit application for exploring student data and making predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD AND CACHE DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load cleaned datasets"""
    try:
        math_df = pd.read_csv('data/math_cleaned_data.csv')
        portuguese_df = pd.read_csv('data/portuguese_cleaned_data.csv')
        return math_df, portuguese_df
    except FileNotFoundError:
        st.error("❌ Data files not found. Please ensure math_cleaned_data.csv and portuguese_cleaned_data.csv exist in the data/ folder.")
        return None, None

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    """Train ML models"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_r2 = r2_score(y_test, lr_pred)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'lr_r2': lr_r2,
        'rf_r2': rf_r2,
        'lr_pred': lr_pred,
        'rf_pred': rf_pred,
        'y_test': y_test
    }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load data
    math_df, portuguese_df = load_data()
    
    if math_df is None or portuguese_df is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Select a page:", [
        "🏠 Home",
        "📈 Data Overview",
        "🔍 Exploratory Analysis",
        "🎯 Math Subject Analysis",
        "🎯 Portuguese Subject Analysis",
        "🤖 ML Model Predictions",
        "💡 Insights & Recommendations"
    ])
    
    if page == "🏠 Home":
        show_home()
    elif page == "📈 Data Overview":
        show_data_overview(math_df, portuguese_df)
    elif page == "🔍 Exploratory Analysis":
        show_eda(math_df, portuguese_df)
    elif page == "🎯 Math Subject Analysis":
        show_subject_analysis(math_df, "Math")
    elif page == "🎯 Portuguese Subject Analysis":
        show_subject_analysis(portuguese_df, "Portuguese")
    elif page == "🤖 ML Model Predictions":
        show_ml_predictions(math_df, portuguese_df)
    elif page == "💡 Insights & Recommendations":
        show_insights()

# ============================================================================
# PAGE 1: HOME
# ============================================================================

def show_home():
    st.title("🎓 Student Performance Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Welcome! 👋
        
        This interactive dashboard provides comprehensive analysis of student 
        performance across **Math** and **Portuguese** subjects.
        
        **Key Features:**
        - 📊 Data exploration and visualization
        - 🔍 Subject-specific analysis
        - 🤖 Machine learning predictions
        - 💡 Actionable insights and recommendations
        
        **Dataset Information:**
        - Multiple student records
        - Academic and lifestyle factors
        - Performance metrics and grades
        """)
    
    with col2:
        st.markdown("""
        ### How to Use 📖
        
        1. **Data Overview** - Get summary statistics
        2. **Exploratory Analysis** - Visualize distributions
        3. **Subject Analysis** - Dive deep into Math/Portuguese
        4. **ML Predictions** - See predictive models
        5. **Insights** - Get recommendations
        
        ### Navigation 🧭
        
        Use the sidebar to navigate between pages.
        Each page has interactive elements you can explore!
        """)
    
    # Display key metrics
    st.markdown("---")
    st.subheader("📊 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Math Students", len(pd.read_csv('data/math_cleaned_data.csv')))
    with col2:
        st.metric("Portuguese Students", len(pd.read_csv('data/portuguese_cleaned_data.csv')))
    with col3:
        math_avg = pd.read_csv('data/math_cleaned_data.csv')['G3'].mean()
        st.metric("Avg Math Grade", f"{math_avg:.2f}")
    with col4:
        por_avg = pd.read_csv('data/portuguese_cleaned_data.csv')['G3'].mean()
        st.metric("Avg Portuguese Grade", f"{por_avg:.2f}")

# ============================================================================
# PAGE 2: DATA OVERVIEW
# ============================================================================

def show_data_overview(math_df, portuguese_df):
    st.title("📈 Data Overview")
    st.markdown("---")
    
    # Subject selector
    subject = st.selectbox("Select Subject:", ["Math", "Portuguese"])
    df = math_df if subject == "Math" else portuguese_df
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Data Completeness", f"{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%")
    
    # Grade Statistics
    st.subheader("📈 Grade Distribution (G3)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Grade", f"{df['G3'].mean():.2f}")
    with col2:
        st.metric("Median Grade", f"{df['G3'].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df['G3'].std():.2f}")
    with col4:
        st.metric("Range", f"{df['G3'].min():.0f} - {df['G3'].max():.0f}")
    
    # Grade categories
    st.markdown("---")
    st.subheader("🎯 Student Performance Categories")
    
    high_achievers = (df['G3'] >= 15).sum()
    average = ((df['G3'] >= 10) & (df['G3'] < 15)).sum()
    struggling = (df['G3'] < 10).sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌟 High Achievers (G3 ≥ 15)", f"{high_achievers} ({high_achievers/len(df)*100:.1f}%)")
    with col2:
        st.metric("📊 Average (10 ≤ G3 < 15)", f"{average} ({average/len(df)*100:.1f}%)")
    with col3:
        st.metric("⚠️ Struggling (G3 < 10)", f"{struggling} ({struggling/len(df)*100:.1f}%)")
    
    # Detailed Statistics Table
    st.markdown("---")
    st.subheader("📋 Detailed Statistics")
    
    stats_display = df.describe().T
    st.dataframe(stats_display, use_container_width=True)
    
    # Data sample
    st.markdown("---")
    st.subheader("👀 Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS
# ============================================================================

def show_eda(math_df, portuguese_df):
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")
    
    subject = st.selectbox("Select Subject:", ["Math", "Portuguese"])
    df = math_df if subject == "Math" else portuguese_df
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Relationships", "Comparisons"])
    
    # TAB 1: Distributions
    with tab1:
        st.subheader("📊 Grade Distribution")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(df['G3'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Final Grade (G3)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{subject} - Final Grade Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        
        # Other distributions
        st.subheader("📈 Key Numeric Features Distribution")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("Select feature:", numeric_cols)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(df[selected_col], bins=20, color='coral', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(selected_col, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title(f'Distribution of {selected_col}', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df[selected_col])
        axes[1].set_ylabel(selected_col, fontweight='bold')
        axes[1].set_title(f'Box Plot of {selected_col}', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
    
    # TAB 2: Correlations
    with tab2:
        st.subheader("🔗 Correlation Heatmap")
        
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{subject} - Correlation Heatmap', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        
        # Top correlations with G3
        st.subheader("⭐ Top Correlations with Final Grade (G3)")
        
        g3_corr = corr_matrix['G3'].sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive Correlations:**")
            for feature, corr in g3_corr.head(5).items():
                if feature != 'G3':
                    st.write(f"  • {feature}: {corr:.3f}")
        
        with col2:
            st.write("**Negative Correlations:**")
            for feature, corr in g3_corr.tail(5).items():
                st.write(f"  • {feature}: {corr:.3f}")
    
    # TAB 3: Relationships
    with tab3:
        st.subheader("📊 Variable vs Final Grade (G3)")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_var = st.selectbox("Select variable:", [c for c in numeric_cols if c != 'G3'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.scatter(df[selected_var], df['G3'], alpha=0.6, s=50, color='steelblue', edgecolors='navy')
        
        # Add trendline
        z = np.polyfit(df[selected_var].dropna(), df['G3'].dropna(), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df[selected_var].min(), df[selected_var].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend Line')
        
        corr = df[selected_var].corr(df['G3'])
        ax.set_xlabel(selected_var, fontweight='bold', fontsize=12)
        ax.set_ylabel('Final Grade (G3)', fontweight='bold', fontsize=12)
        ax.set_title(f'{selected_var} vs Final Grade (Correlation: {corr:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
    
    # TAB 4: Comparisons
    with tab4:
        st.subheader("📊 Grade Distribution by Category")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            selected_cat = st.selectbox("Select category:", categorical_cols)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            df.boxplot(column='G3', by=selected_cat, ax=ax)
            ax.set_xlabel(selected_cat, fontweight='bold')
            ax.set_ylabel('Final Grade (G3)', fontweight='bold')
            ax.set_title(f'Final Grade Distribution by {selected_cat}', fontweight='bold')
            
            plt.suptitle('')
            st.pyplot(fig)

# ============================================================================
# PAGE 4 & 5: SUBJECT ANALYSIS
# ============================================================================

def show_subject_analysis(df, subject):
    st.title(f"🎯 {subject} Subject Analysis")
    st.markdown("---")
    
    # Variable groups
    variable_groups = {
        'Academic Behavior': ['studytime', 'failures', 'absences', 'G1', 'G2'],
        'Family Background': ['Medu', 'Fedu', 'famsize', 'famsup'],
        'Lifestyle & Social': ['freetime', 'goout', 'Dalc', 'Walc', 'romantic'],
        'School Support': ['schoolsup', 'paid', 'internet'],
    }
    
    # Select variable group
    selected_group = st.selectbox("Select Variable Group:", list(variable_groups.keys()))
    selected_vars = variable_groups[selected_group]
    
    st.subheader(f"Analysis of {selected_group}")
    
    # Create visualizations for the group
    valid_vars = [v for v in selected_vars if v in df.columns]
    
    if len(valid_vars) > 0:
        # Row 1: Distributions
        st.markdown("### 📊 Feature Distributions")
        
        cols = st.columns(2)
        for idx, var in enumerate(valid_vars[:2]):
            with cols[idx % 2]:
                fig, ax = plt.subplots(figsize=(8, 4))
                
                if df[var].dtype in ['float64', 'int64']:
                    ax.hist(df[var], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
                    ax.set_title(f'Distribution of {var}', fontweight='bold')
                else:
                    df[var].value_counts().plot(kind='bar', ax=ax, color='coral', alpha=0.7)
                    ax.set_title(f'Distribution of {var}', fontweight='bold')
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
                ax.set_ylabel('Frequency', fontweight='bold')
                st.pyplot(fig)
        
        # Row 2: Correlations with G3
        st.markdown("### 🔗 Impact on Final Grade")
        
        numeric_vars = [v for v in valid_vars if df[v].dtype in ['float64', 'int64']]
        
        cols = st.columns(2)
        for idx, var in enumerate(numeric_vars[:2]):
            with cols[idx % 2]:
                fig, ax = plt.subplots(figsize=(8, 5))
                
                mask = ~(df[var].isna() | df['G3'].isna())
                ax.scatter(df.loc[mask, var], df.loc[mask, 'G3'], alpha=0.6, 
                         s=30, color='steelblue', edgecolors='navy')
                
                z = np.polyfit(df.loc[mask, var], df.loc[mask, 'G3'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df.loc[mask, var].min(), df.loc[mask, var].max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", linewidth=2)
                
                corr = df.loc[mask, var].corr(df.loc[mask, 'G3'])
                ax.set_xlabel(var, fontweight='bold')
                ax.set_ylabel('Final Grade (G3)', fontweight='bold')
                ax.set_title(f'{var} vs G3 (r={corr:.3f})', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
    
    # Summary Statistics
    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    
    summary_stats = []
    for var in numeric_vars:
        if var in df.columns:
            summary_stats.append({
                'Variable': var,
                'Mean': f"{df[var].mean():.2f}",
                'Std Dev': f"{df[var].std():.2f}",
                'Min': f"{df[var].min():.2f}",
                'Max': f"{df[var].max():.2f}",
                'Correlation with G3': f"{df[var].corr(df['G3']):.3f}"
            })
    
    if summary_stats:
        st.dataframe(pd.DataFrame(summary_stats), use_container_width=True)

# ============================================================================
# PAGE 6: ML PREDICTIONS
# ============================================================================

def show_ml_predictions(math_df, portuguese_df):
    st.title("🤖 Machine Learning Predictions")
    st.markdown("---")
    
    # Subject selector
    subject = st.radio("Select Subject:", ["Math", "Portuguese"], horizontal=True)
    df = math_df if subject == "Math" else portuguese_df
    
    # Prepare data
    y = df['G3']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['G3', 'G1', 'G2']]
    
    X = df[numeric_cols].copy()
    
    # Encode categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
    
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    model_results = train_models(X_train, X_test, y_train, y_test)
    
    # Display results
    st.subheader("📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Linear Regression R² Score", f"{model_results['lr_r2']:.4f}")
    with col2:
        st.metric("Random Forest R² Score", f"{model_results['rf_r2']:.4f}")
    
    # Predictions vs Actual
    st.markdown("---")
    st.subheader("📈 Predictions vs Actual Values")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Linear Regression")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(model_results['y_test'], model_results['lr_pred'], alpha=0.6, s=30)
        ax.plot([model_results['y_test'].min(), model_results['y_test'].max()], 
               [model_results['y_test'].min(), model_results['y_test'].max()], 'r--', lw=2)
        ax.set_xlabel('Actual Grade', fontweight='bold')
        ax.set_ylabel('Predicted Grade', fontweight='bold')
        ax.set_title('Linear Regression Predictions', fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.write("### Random Forest")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(model_results['y_test'], model_results['rf_pred'], alpha=0.6, s=30, color='coral')
        ax.plot([model_results['y_test'].min(), model_results['y_test'].max()], 
               [model_results['y_test'].min(), model_results['y_test'].max()], 'r--', lw=2)
        ax.set_xlabel('Actual Grade', fontweight='bold')
        ax.set_ylabel('Predicted Grade', fontweight='bold')
        ax.set_title('Random Forest Predictions', fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Feature Importance
    st.markdown("---")
    st.subheader("🎯 Feature Importance (Random Forest)")
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model_results['rf_model'].feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(feature_importance)), feature_importance['Importance'].values, color='steelblue')
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['Feature'].values)
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title(f'Top 15 Features Predicting {subject} Grades', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    
    # Feature importance table
    st.dataframe(feature_importance.reset_index(drop=True), use_container_width=True)

# ============================================================================
# PAGE 7: INSIGHTS & RECOMMENDATIONS
# ============================================================================

def show_insights():
    st.title("💡 Insights & Recommendations")
    st.markdown("---")
    
    # Load data for analysis
    math_df = pd.read_csv('data/math_cleaned_data.csv')
    portuguese_df = pd.read_csv('data/portuguese_cleaned_data.csv')
    
    tab1, tab2, tab3 = st.tabs(["Key Findings", "Recommendations", "Best Practices"])
    
    with tab1:
        st.subheader("📌 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Math Subject")
            math_high = (math_df['G3'] >= 15).sum() / len(math_df) * 100
            math_struggling = (math_df['G3'] < 10).sum() / len(math_df) * 100
            
            st.write(f"""
            - **High Achievers:** {math_high:.1f}%
            - **Struggling Students:** {math_struggling:.1f}%
            - **Average Grade:** {math_df['G3'].mean():.2f}
            - **Grade Volatility:** {math_df['G3'].std():.2f}
            """)
        
        with col2:
            st.markdown("### Portuguese Subject")
            por_high = (portuguese_df['G3'] >= 15).sum() / len(portuguese_df) * 100
            por_struggling = (portuguese_df['G3'] < 10).sum() / len(portuguese_df) * 100
            
            st.write(f"""
            - **High Achievers:** {por_high:.1f}%
            - **Struggling Students:** {por_struggling:.1f}%
            - **Average Grade:** {portuguese_df['G3'].mean():.2f}
            - **Grade Volatility:** {portuguese_df['G3'].std():.2f}
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 🔍 Analysis Summary
        
        1. **Study Time Impact**: Students with higher study time consistently achieve better grades
        2. **Attendance Matters**: Absence rates show strong negative correlation with performance
        3. **Family Support**: Parental education and family support play significant roles
        4. **Social Factors**: Balance between social life and academics is crucial
        5. **Previous Performance**: Past grades (G1, G2) are strong predictors of final grades
        """)
    
    with tab2:
        st.subheader("🎯 Recommendations for Students")
        
        st.markdown("""
        ### For Struggling Students (G3 < 10):
        
        1. **Increase Study Time**
           - Start with 1-2 hours daily, gradually increase to 3-4 hours
           - Use structured study sessions with breaks
        
        2. **Improve Attendance**
           - Aim for 90%+ attendance rate
           - Set reminders and schedule classes
        
        3. **Seek Support**
           - Utilize school tutoring services
           - Join study groups with peers
           - Talk to teachers about challenges
        
        4. **Manage Distractions**
           - Reduce social outings during exam periods
           - Limit recreational screen time
           - Create dedicated study space
        
        ### For Average Students (10 ≤ G3 < 15):
        
        1. **Consistency is Key**
           - Maintain regular study schedule
           - Review notes after each class
           - Complete assignments on time
        
        2. **Set Goals**
           - Target specific improvements in weak areas
           - Track progress regularly
           - Celebrate milestones
        
        3. **Enhance Learning**
           - Try different study techniques
           - Use active recall and spaced repetition
           - Find study partners
        
        ### For High Achievers (G3 ≥ 15):
        
        1. **Maintain Success**
           - Continue current study habits
           - Help other students (peer tutoring)
           - Explore advanced topics
        
        2. **Develop Leadership**
           - Join academic clubs
           - Mentor struggling students
           - Take on leadership roles
        """)
    
    with tab3:
        st.subheader("📚 Best Practices for Success")
        
        st.markdown("""
        ### Study Habits
        
        ✅ **DO:**
        - Study in regular, focused sessions (25-50 minute blocks)
        - Review notes within 24 hours of class
        - Use active learning techniques
        - Take practice tests
        - Get 7-9 hours of sleep
        
        ❌ **DON'T:**
        - Cram all night before exams
        - Passive reading without notes
        - Study while tired or distracted
        - Skip classes
        - Over-rely on alcohol/stimulants
        
        ### Time Management
        
        - **Plan Ahead:** Use calendar/planner for assignments and exams
        - **Prioritize:** Focus on high-impact tasks first
        - **Balance:** Mix study with physical activity and social time
        - **Flexibility:** Adjust plans based on progress
        
        ### Support Systems
        
        - **Family Support:** Keep family informed of progress
        - **Teachers:** Attend office hours, ask for help
        - **Peers:** Form study groups, share resources
        - **School Services:** Use tutoring, counseling, health services
        
        ### Health & Wellness
        
        - **Sleep:** Prioritize 7-9 hours nightly
        - **Exercise:** 30 minutes daily physical activity
        - **Nutrition:** Eat balanced meals, stay hydrated
        - **Mental Health:** Manage stress, seek help if needed
        """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()