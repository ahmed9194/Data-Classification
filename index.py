import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data", header=None)
df.columns = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", 
    "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"
]

# Encode class labels
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])  # g -> 1, h -> 0

# Balance the data
gamma = df[df['class'] == 1]
hadron = df[df['class'] == 0]

if len(gamma) > len(hadron):
    gamma_balanced = gamma.sample(n=len(hadron), random_state=42)
    df_balanced = pd.concat([gamma_balanced, hadron], axis=0)
elif len(hadron) > len(gamma):
    hadron_balanced = hadron.sample(n=len(gamma), random_state=42)
    df_balanced = pd.concat([gamma, hadron_balanced], axis=0)
else:
    df_balanced = df.copy()

df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle

# Split data
X = df_balanced.drop('class', axis=1)
y = df_balanced['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Helper function for evaluation
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name} Results")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return acc, prec, rec, f1

# 1. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
evaluate_model("Decision Tree", dt, X_test, y_test)

# 2. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
evaluate_model("Naive Bayes", nb, X_test, y_test)

# 3. Random Forest (tuning n_estimators)
rf_params = {'n_estimators': [10, 50]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
evaluate_model("Random Forest", best_rf, X_test, y_test)

# 4. AdaBoost (tuning n_estimators)
ab_params = {'n_estimators': [10, 50]}
ab_grid = GridSearchCV(AdaBoostClassifier(random_state=42), ab_params, cv=3)
ab_grid.fit(X_train, y_train)
best_ab = ab_grid.best_estimator_
evaluate_model("AdaBoost", best_ab, X_test, y_test)

# Summary Table
models = [dt, nb, best_rf, best_ab]
names = ["Decision Tree", "Naive Bayes", "Random Forest", "AdaBoost"]
results = [evaluate_model(name, model, X_test, y_test) for name, model in zip(names, models)]
summary = pd.DataFrame(results, columns=["Accuracy", "Precision", "Recall", "F1-Score"], index=names)
print("\nComparison Table:")
print(summary)

