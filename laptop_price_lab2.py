import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# ============================================================
#   Lab 2: Model Comparison and Evaluation
#   Charlie C. Omongos
#   IT3R12
#   IT325 Machine Learning | Laptop Price Dataset
#   Models: Linear Regression, Logistic Regression, KNN,
#           Decision Tree, Naive Bayes
# ============================================================

print("=" * 60)
print("  Lab 2: Model Comparison and Evaluation")
print("  IT325 Machine Learning | Laptop Price Dataset")
print("=" * 60)

# --- SECTION 1: Loading the Dataset ---

print("\n--- SECTION 1: Loading the Dataset ---")

df = pd.read_csv('laptop_price - dataset.csv')

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# --- SECTION 2: Converting Price to Categories (Target Variable) ---

print("\n--- SECTION 2: Converting Price to Categories ---")

bins = [0, 700, 1500, float('inf')]
labels = ['Budget', 'Mid-range', 'Premium']

df['Price_Category'] = pd.cut(df['Price (Euro)'], bins=bins, labels=labels)

print("\nPrice Category Distribution:")
print(df['Price_Category'].value_counts())

df['Price_Category'].value_counts().plot(
    kind='bar',
    color=['steelblue', 'coral', 'mediumseagreen'],
    edgecolor='black'
)

plt.title('Price Category Distribution')
plt.xlabel('Price Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- SECTION 3: Data Preparation ---

print("\n--- SECTION 3: Data Preparation ---")

# Drop columns that are not useful as features
df_model = df.drop(columns=['Product', 'ScreenResolution', 'Memory', 'Price (Euro)'])

y_raw = df_model['Price_Category']
df_features = df_model.drop(columns=['Price_Category'])

# Encode categorical columns
cat_cols = df_features.select_dtypes(include='object').columns.tolist()

print("Categorical columns to encode:", cat_cols)

le = LabelEncoder()

for col in cat_cols:
    df_features[col] = le.fit_transform(df_features[col].astype(str))

# Encode the target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw.astype(str))

print("Encoded target classes:", le_target.classes_)

X = df_features

print("\nFeature shape:", X.shape)
print("Target shape :", y.shape)

# --- SECTION 4: Train-Test Split ---

print("\n--- SECTION 4: Train-Test Split ---")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size : {X_test.shape[0]} rows")

# --- SECTION 5: Feature Scaling ---
# KNN and Logistic Regression benefit from scaled features.
# Scaling all features so that every model uses the same prepared data.

print("\n--- SECTION 5: Feature Scaling ---")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling applied using StandardScaler.")

# --- SECTION 6: Training All Five Models ---

print("\n--- SECTION 6: Training All Five Models ---")

# Linear Regression is a regression model but we adapt it for classification
# by rounding predictions to the nearest class label.

models = {
    'Linear Regression': LinearRegression(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

results = {}

for model_name, model in models.items():
    print(f"  Training {model_name}...", end=' ')

    model.fit(X_train_scaled, y_train)

    if model_name == 'Linear Regression':
        # Linear Regression outputs continuous values.
        # Round and clip to valid class indices for classification.
        y_pred_raw = model.predict(X_test_scaled)
        y_pred = np.clip(np.round(y_pred_raw).astype(int), 0, len(le_target.classes_) - 1)
    else:
        y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    rec = recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    results[model_name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'y_pred': y_pred
    }

    print(f"Done. (Accuracy: {acc*100:.2f}%)")

print("\nAll five models trained successfully!")

# --- SECTION 7: Model Evaluation Results ---

print("\n" + "=" * 60)
print("  SECTION 7: Model Evaluation Results")
print("=" * 60)

for model_name, res in results.items():

    print(f"\n{'='*50}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*50}")

    print(f"  Accuracy  : {res['Accuracy']:.4f} ({res['Accuracy']*100:.2f}%)")
    print(f"  Precision : {res['Precision']:.4f}")
    print(f"  Recall    : {res['Recall']:.4f}")
    print(f"  F1 Score  : {res['F1 Score']:.4f}")

    print(f"\n  Classification Report:")

    print(
        classification_report(
            y_test,
            res['y_pred'],
            target_names=le_target.classes_,
            zero_division=0
        )
    )

    cm = confusion_matrix(y_test, res['y_pred'])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=le_target.classes_
    )

    disp.plot(cmap='Blues')

    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.show()

# --- SECTION 8: Model Comparison Summary ---

print("\n" + "=" * 60)
print("  SECTION 8: Model Comparison Summary")
print("=" * 60)

comparison_data = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

for model_name, res in results.items():

    comparison_data['Model'].append(model_name)
    comparison_data['Accuracy'].append(round(res['Accuracy'], 4))
    comparison_data['Precision'].append(round(res['Precision'], 4))
    comparison_data['Recall'].append(round(res['Recall'], 4))
    comparison_data['F1 Score'].append(round(res['F1 Score'], 4))

comparison_df = pd.DataFrame(comparison_data)

comparison_df = comparison_df.set_index('Model')

print("\n")
print(comparison_df.to_string())

# --- SECTION 9: Plotting Model Comparison Bar Chart ---

print("\n--- SECTION 9: Plotting Model Comparison Bar Chart ---")

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

model_names = list(results.keys())

x = np.arange(len(model_names))

width = 0.18

colors = [
    'steelblue',
    'coral',
    'mediumseagreen',
    'mediumpurple'
]

fig, ax = plt.subplots(figsize=(14, 6))

for i, metric in enumerate(metrics):

    values = [results[m][metric] for m in model_names]

    ax.bar(
        x + i * width,
        values,
        width=width,
        label=metric,
        color=colors[i],
        edgecolor='black'
    )

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Comparison - All Metrics (5 Models)')

ax.set_xticks(x + width * 1.5)

ax.set_xticklabels(
    model_names,
    rotation=12,
    ha='right'
)

ax.set_ylim(0, 1.1)

ax.legend()

ax.grid(
    axis='y',
    linestyle='--',
    alpha=0.5
)

plt.tight_layout()
plt.show()

# --- SECTION 10: Analysis and Findings ---

print("\n" + "=" * 60)
print("  SECTION 10: Analysis and Findings")
print("=" * 60)

best_accuracy = comparison_df['Accuracy'].idxmax()
best_recall = comparison_df['Recall'].idxmax()
best_f1 = comparison_df['F1 Score'].idxmax()

print(f"\n  Model with Highest Accuracy  : {best_accuracy}")
print(f"  Model with Highest Recall    : {best_recall}")
print(f"  Model with Most Balanced F1  : {best_f1}")

# --- SECTION 11: Recommended Models for Tuning ---

print("\n" + "=" * 60)
print("  SECTION 11: Top Two Models for Tuning Stage")
print("=" * 60)

# Rank models by F1 Score (most balanced metric)
ranked = comparison_df.sort_values('F1 Score', ascending=False)
top_two = ranked.index[:2].tolist()

print(f"\n  1st: {top_two[0]}")
print(f"  2nd: {top_two[1]}")

print(f"""
  Selection Rationale:
  {top_two[0]} and {top_two[1]} are selected for the
  tuning stage because they achieved the highest overall F1 Scores
  among all five models. The F1 Score is a balanced metric that
  considers both precision and recall, making it more reliable
  than accuracy alone, especially when the class distribution
  is imbalanced. These two models demonstrated the strongest
  generalization ability on the unseen test set and are the
  most promising candidates for hyperparameter optimization.
""")

# --- SECTION 12: Reflection ---

print("=" * 60)
print("  SECTION 12: Reflection")
print("=" * 60)

print("""
  Reflection Summary:

  Linear Regression, while primarily a regression algorithm,
  was adapted for classification by rounding continuous outputs
  to discrete class labels. Its performance is expectedly lower
  compared to dedicated classifiers, confirming that specialized
  classification algorithms are better suited for categorical
  prediction tasks.

  Logistic Regression and KNN both benefit from feature scaling,
  which was applied using StandardScaler. The comparison shows
  that model choice significantly affects performance, and
  evaluation must go beyond accuracy to include precision,
  recall, and F1 Score for a fair and meaningful assessment.
""")

# --- FINAL SUMMARY ---

print("=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)

print("\nComparison Table:")
print(comparison_df.to_string())

print(f"""
  Best Accuracy  -> {best_accuracy}
  Best Recall    -> {best_recall}
  Best F1 Score  -> {best_f1}

  Top Two Models for Tuning:
  -> {top_two[0]}
  -> {top_two[1]}

""")

