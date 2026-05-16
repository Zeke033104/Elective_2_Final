# Laptop Price Prediction — IT325 Machine Learning

**Student:** Charlie C. Omongos  
**Section:** IT3R12  
**Course:** IT325 — Machine Learning (Elective 2)

---

## 📌 Project Overview

This repository contains all the Python scripts developed across **Module 3, Module 4, and Module 5 (Lab 2 — Final Activity)** for the IT325 Machine Learning course. Each script builds upon the previous one, progressively exploring and modeling the **Laptop Price dataset** (`laptop_price - dataset.csv`) to predict laptop price categories using various machine learning algorithms.

---

## 📁 Project Structure

```
├── laptop_price - dataset.csv      # Dataset used across all activities
├── laptop_price.py                 # Module 3 Activity — EDA
├── laptop_price_activity4.py       # Module 4 Activity — Model Comparison
├── laptop_price_lab2.py            # Module 5 / Lab 2 — Final Activity
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 📝 File Descriptions

### 1. `laptop_price.py` — Module 3 Activity (Exploratory Data Analysis)

This script performs **Exploratory Data Analysis (EDA)** on the Laptop Price dataset. It is the foundation of the project and focuses on understanding the data before any modeling.

**What it does:**
- Loads the dataset and inspects its shape, column names, and data types
- Checks for missing values and duplicate rows
- Generates descriptive statistics for the `Price (Euro)` column
- Produces visualizations:
  - Histogram of price distribution
  - Bar chart of laptop count by company
  - Bar chart of laptop type distribution
  - Bar chart of RAM distribution
  - Pie chart of CPU company market share
  - Box plot of price by RAM size
- Includes written observations and interpretations of the data

---

### 2. `laptop_price_activity4.py` — Module 4 Activity (Model Comparison)

This script builds on the EDA from Module 3 and introduces **classification modeling**. It trains and evaluates **five machine learning models** to classify laptops into price categories (Budget, Mid-range, Premium).

**Models trained:**
| # | Algorithm              |
|---|------------------------|
| 1 | Logistic Regression    |
| 2 | Decision Tree          |
| 3 | K-Nearest Neighbors    |
| 4 | Naive Bayes            |
| 5 | Gradient Boosting      |

**What it does:**
- Converts the continuous `Price (Euro)` column into three categories: *Budget*, *Mid-range*, *Premium*
- Encodes categorical features and the target variable using `LabelEncoder`
- Splits the data into 80% training / 20% testing sets
- Trains all five models and evaluates each using:
  - Accuracy, Precision, Recall, F1 Score
  - Classification Report
  - Confusion Matrix (with visualization)
- Produces a side-by-side bar chart comparing all models across all metrics
- Identifies the best model and provides a recommendation

---

### 3. `laptop_price_lab2.py` — Module 5 / Lab 2 Final Activity (Comprehensive Model Evaluation)

This is the **final activity** that consolidates and extends the work from all prior modules. It trains and tests **all five models** developed throughout the course, applying proper preprocessing (including **feature scaling with StandardScaler**) and providing a comprehensive comparative analysis.

**Models trained:**
| # | Algorithm              |
|---|------------------------|
| 1 | Linear Regression *    |
| 2 | Logistic Regression    |
| 3 | K-Nearest Neighbors    |
| 4 | Decision Tree          |
| 5 | Naive Bayes            |

> \* Linear Regression is a regression algorithm adapted for classification by rounding and clipping its continuous output to the nearest valid class label.

**What it does:**
- Loads and prepares the dataset (same binning and encoding strategy)
- Applies **StandardScaler** for feature scaling (benefits KNN and Logistic Regression)
- Trains all five models on the scaled features
- Evaluates each model with full classification reports and confusion matrices
- Compares all models in a summary table and grouped bar chart
- Identifies the best-performing model by Accuracy, Recall, and F1 Score
- Ranks the **top two models** recommended for hyperparameter tuning
- Includes a detailed reflection section

---

## 📊 Dataset

**File:** `laptop_price - dataset.csv`

- **Rows:** 1,275 laptop entries
- **Columns:** 15 features including Company, TypeName, RAM, CPU, GPU, OS, Weight, Screen size, and Price
- **Target Variable:** `Price (Euro)` → binned into `Budget`, `Mid-range`, `Premium`
- No missing values or duplicate rows

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python 3.8+** installed on your system

### Step 1: Clone the Repository

```bash
git clone https://github.com/Zeke033104/Elective_2_Final.git
cd Elective_2_Final
```

### Step 2: (Optional) Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the following packages:

| Package        | Purpose                                           |
|----------------|---------------------------------------------------|
| `pandas`       | Data loading, manipulation, and analysis          |
| `numpy`        | Numerical operations and array handling            |
| `matplotlib`   | Data visualization (charts, plots, confusion matrices) |
| `scikit-learn` | Machine learning models, preprocessing, and metrics |

### Step 4: Run the Scripts

Run each script individually using Python:

```bash
# Module 3 — Exploratory Data Analysis
python laptop_price.py

# Module 4 — Model Comparison (5 classification models)
python laptop_price_activity4.py

# Module 5 / Lab 2 — Final Activity (comprehensive evaluation)
python laptop_price_lab2.py
```

> **Note:** Each script will display visualizations (charts and plots) in pop-up windows. Close each plot window to continue execution.

---

## 🧪 Summary of Models Used Across All Activities

| Model                | Module 3 | Module 4 | Lab 2 (Final) |
|----------------------|:--------:|:--------:|:--------------:|
| Linear Regression    |    —     |    —     |       ✅       |
| Logistic Regression  |    —     |    ✅    |       ✅       |
| K-Nearest Neighbors  |    —     |    ✅    |       ✅       |
| Decision Tree        |    —     |    ✅    |       ✅       |
| Naive Bayes          |    —     |    ✅    |       ✅       |
| Gradient Boosting    |    —     |    ✅    |       —        |

---
