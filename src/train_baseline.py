import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

os.makedirs("models/baseline", exist_ok=True)

# Load Data
data = pd.read_csv("data/features.csv")

print("\n-------------------------")
print("Dataset shape:", data.shape)

# Prepare Features + Target
TARGET = "risk_label"

X = data.drop(
    columns=["Date", TARGET, "peak_price"]
)

y = data[TARGET].shift(-1)

X = X.iloc[:-1]
y = y.dropna()


# Time-Based Train/Test Split
split = int(len(X) * 0.75)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTrain:", X_train.shape)
print("Test:", X_test.shape)

print(
    "\nTrain dates:", data["Date"].iloc[0],
    "to",
    data["Date"].iloc[split]
)

print(
    "Test dates:", data["Date"].iloc[split],
    "to",
    data["Date"].iloc[-1]
)

# Evaluation
def evaluate(name, model):

    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]

    # Lower threshold to prioritize risk detection
    pred = (prob >= 0.35).astype(int)

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1 Score": round(f1_score(y_test, pred), 4),
        "MCC": round(matthews_corrcoef(y_test, pred), 4)
    }

    return metrics, model

# Baseline Models
models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": HistGradientBoostingClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss"
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=True
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}

results_list = []
baseline_models = {}
print('\n')

for name, model in models.items():

    print(f"Training {name}...")
    metrics, fitted_model = evaluate(
        name,
        model
    )

    results_list.append(metrics)
    baseline_models[name] = fitted_model

# Baseline Results
results_df = pd.DataFrame(results_list)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

print("\n================ BASELINE RESULTS ================\n")

print(results_df.to_string(index=False))

# Save Baseline Models
for _, row in results_df.iterrows():

    model_name = row["Model"]
    clean_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    file_name = (
        f"models/baseline/"
        f"{clean_name}_baseline.pkl"
    )

    joblib.dump(
        baseline_models[model_name],
        file_name
    )

print("\nAll baseline models saved!")
print("--------------------------------------------------")