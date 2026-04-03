# 📊 Student Performance Analysis (Math vs Portuguese)

## 🚀 Overview

This project presents a comprehensive analysis of student performance using the **UCI Student Performance Dataset**, focusing on **Mathematics and Portuguese subjects**.

It combines:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Statistical insights
* Machine Learning models
* Interactive dashboard

The goal is to understand:

> ❗ *How different factors affect student performance, and how learning patterns differ across subjects.*

---

## 🎯 Key Features

* 📥 **Data Pipeline** – Load, clean, and validate raw data
* 📊 **EDA** – Visualize patterns and relationships
* 🔍 **Subject-wise Analysis** – Deep dive into Math & Portuguese separately
* ⚖️ **Comparative Analysis** – Identify differences in learning behavior
* 🤖 **Machine Learning** – Predict final grades (G3)
* 🎨 **Interactive Dashboard** – Clean UI with real-time insights

---

## 📁 Project Structure

```
student-performance-analysis/
│
├── data/
│   ├── student-mat.csv
│   ├── student-por.csv
│   ├── math_cleaned_data.csv
│   ├── portuguese_cleaned_data.csv
│   └── cleaned_student_data.csv
│
├── research_output/
│   ├── math_analysis/
│   ├── portuguese_analysis/
│   ├── comparative_analysis/
│   └── ml_analysis/
│
├── outputs/
│   └── *.png
│
├── load_data.py
├── prepare_data.py
├── eda_analysis.py
├── math_analysis_grouping.py
├── portuguese_analysis_grouping.py
├── subject_comparison.py
├── contrastive_analysis.py
├── ml_model_prediction.py
├── dashboard_2.py
│
└── README.md
```

---

## 🔍 Workflow

### 1️⃣ Data Processing

* Load raw datasets
* Clean missing/invalid values
* Convert data types
* Validate ranges

---

### 2️⃣ Exploratory Data Analysis (EDA)

* Grade distributions
* Studytime vs performance
* Absences vs performance
* Correlation heatmaps

---

### 3️⃣ Subject-wise Analysis

Separate deep analysis for:

* 📘 Mathematics
* 📗 Portuguese

Includes:

* Academic behavior
* Lifestyle factors
* Family background
* Support systems

---

### 4️⃣ Comparative Analysis

Compare patterns across subjects:

* Studytime impact
* Absences impact
* Social behavior differences
* Correlation differences

---

### 5️⃣ Machine Learning

Models used:

* Linear Regression
* Random Forest

🎯 Goal:
Predict **G3 (final grade)**

📊 Outputs:

* R² Score
* RMSE
* MAE
* Feature importance

---

### 6️⃣ Interactive Dashboard

Run:

```bash
streamlit run dashboard_minimalist.py
```

Then open:

```
http://localhost:8501
```

#### Dashboard Pages:

* **Overview** → summary metrics
* **Analysis** → distributions & correlations
* **Predictions** → ML results
* **Insights** → conclusions & recommendations

---

## ▶️ How to Run

### Step 1 — Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit scipy
```

---

### Step 2 — Run pipeline

```bash
python load_data.py
python prepare_data.py
python eda_analysis.py
python math_analysis_grouping.py
python portuguese_analysis_grouping.py
python subject_comparison.py
python contrastive_analysis.py
python ml_model_prediction.py
```

---

### Step 3 — Launch dashboard

```bash
python -m streamlit run dashboard_minimalist.py
```

---

## 📊 Data Requirements

* CSV format (semicolon `;` separated)
* Required columns include:

  * G1, G2, G3 (grades)
  * studytime, absences
  * freetime, goout
  * Medu, Fedu
  * schoolsup, famsup

---

## 📈 Outputs

After running all scripts:

* 📊 Visualizations (PNG files)
* 📄 Summary reports (TXT files)
* 🤖 ML results (metrics + feature importance)

---

## 🧠 Key Insights

* Academic performance is influenced by multiple factors
* Study behavior impacts Math more strongly
* Lifestyle factors show varying influence
* Previous grades are strong predictors
* Learning patterns differ across subjects

---

## 🔧 Troubleshooting

### ❌ Streamlit not recognized

```bash
python -m streamlit run dashboard_minimalist.py
```

### ❌ Missing modules

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit scipy
```

### ❌ Dataset not found

* Ensure files exist in `data/` folder

---

## 📚 References

* UCI Machine Learning Repository – Student Performance Dataset
* Pandas, NumPy, Scikit-learn, Streamlit documentation

---

## 👩‍💻 Author

**Vidhi Pratheesh**


---


