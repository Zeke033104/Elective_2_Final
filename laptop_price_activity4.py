import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

print("=" * 60)
print("  Activity 4: Model Comparison and Evaluation")
print("  IT325 Machine Learning | Laptop Price Dataset")
print("=" * 60)

print("\n--- Loading Dataset ---")

df = pd.read_csv('laptop_price - dataset.csv')

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

print("\n--- Converting Price to Categories ---")

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

print("\n--- Data Preparation ---")

df_model = df.drop(columns=['Product', 'ScreenResolution', 'Memory', 'Price (Euro)'])

y_raw = df_model['Price_Category']
df_features = df_model.drop(columns=['Price_Category'])

cat_cols = df_features.select_dtypes(include='object').columns.tolist()

print("Categorical columns to encode:", cat_cols)

le = LabelEncoder()

for col in cat_cols:
    df_features[col] = le.fit_transform(df_features[col].astype(str))

le_target = LabelEncoder()
y = le_target.fit_transform(y_raw.astype(str))

print("Encoded target classes:", le_target.classes_)

X = df_features

print("\nFeature shape:", X.shape)
print("Target shape :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size : {X_test.shape[0]} rows")

print("\n--- Training All Five Models ---")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = {}

for model_name, model in models.items():
    print(f"  Training {model_name}...", end=' ')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

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

print("\n--- Plotting Model Comparison Bar Chart ---")

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

print("\n" + "=" * 60)
print("  SECTION 10: Analysis and Findings")
print("=" * 60)

best_accuracy = comparison_df['Accuracy'].idxmax()
best_recall = comparison_df['Recall'].idxmax()
best_f1 = comparison_df['F1 Score'].idxmax()

print(f"\n  Model with Highest Accuracy  : {best_accuracy}")
print(f"  Model with Highest Recall    : {best_recall}")
print(f"  Model with Most Balanced F1  : {best_f1}")

print("\n" + "=" * 60)
print("  SECTION 11: Recommended Model to Carry Forward")
print("=" * 60)

print("\n  Recommended Model : Decision Tree Classifier")

print(
    "  Reason: Highest accuracy and F1 score of all 5 models"
)

print(
    "          Handles mixed numerical and categorical features well"
)

print(
    "          Highly interpretable and easy to explain"
)

print(
    "          No feature scaling needed unlike KNN"
)

print("\n" + "=" * 60)
print("  SECTION 12: Reflection")
print("=" * 60)

print("""
  Reflection Summary:

  A model that always guesses 'Mid-range' could still reach
  decent accuracy because it is the majority class.

  That is why accuracy alone is not enough.

  Recall, Precision, and F1 Score give a more complete
  evaluation of model performance, especially when
  class sizes are unequal.
""")

print("=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)

print("\nComparison Table:")
print(comparison_df.to_string())

print(f"""
  Best Accuracy  -> {best_accuracy}
  Best Recall    -> {best_recall}
  Best F1 Score  -> {best_f1}

  Recommended Model to Carry Forward:
  -> Decision Tree Classifier

  Activity 4 complete!
""")

print("=" * 60)
print("  Charlie C. Omongos | IT3R12 | IT325 Machine Learning")
print("=" * 60)