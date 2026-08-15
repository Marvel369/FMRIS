import pandas as pd
import joblib
import os

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    ParameterGrid
)
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
    matthews_corrcoef,
    roc_auc_score
)

os.makedirs("models/tuned", exist_ok=True)
data = pd.read_csv("data/features.csv")
TARGET = "risk_label"

X = data.drop(columns=["Date", TARGET, "peak_price"])
y = data[TARGET].shift(-1)

X = X.iloc[:-1]
y = y.dropna()

# TIME-BASED TRAIN / TEST SPLIT
split = int(len(X) * 0.75)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTrain:", X_train.shape)
print("Test:", X_test.shape)

print(
    "\nTrain dates:",
    data["Date"].iloc[0],
    "to",
    data["Date"].iloc[split]
)

print(
    "Test dates:",
    data["Date"].iloc[split],
    "to",
    data["Date"].iloc[-1]
)

# EVALUATION
def evaluate(model_name, model):

    prob = model.predict_proba(X_test)[:, 1]
    # Risk-sensitive threshold
    pred = (prob >= 0.35).astype(int)

    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1 Score": round(f1_score(y_test, pred), 4),
        "MCC": round(matthews_corrcoef(y_test, pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, prob), 4)
    }

# CLASS WEIGHT
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

weight = negative / positive

# TIME SERIES CROSS VALIDATION
tscv = TimeSeriesSplit(n_splits=5)

# MODELS + PARAMETER GRIDS

tuned = {

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "n_estimators": [300, 400],
            "max_depth": [3, 4],
            "min_samples_leaf": [3, 5]
        }
    ),

    "Gradient Boosting": (
        HistGradientBoostingClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "max_iter": [200, 300],
            "learning_rate": [0.03, 0.05],
            "max_depth": [3, 4]
        }
    ),

    "XGBoost": (
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=weight
        ),
        {
            "n_estimators": [200, 300],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.05]
        }
    ),

    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                class_weight="balanced"
            ))
        ]),
        {
            "model__C": [0.1, 1.0],
            "model__gamma": ["scale", "auto"]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "max_depth": [3, 5],
            "min_samples_leaf": [3, 5]
        }
    ),

    "Logistic Regression": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),
        {
            "model__C": [4, 5, 6]
        }
    )
}

# LOAD BASELINE MODELS

baseline_dir = "models/baseline"

baseline_files = {
    "Logistic Regression": "logistic_regression_baseline.pkl",
    "Random Forest": "random_forest_baseline.pkl",
    "Gradient Boosting": "gradient_boosting_baseline.pkl",
    "XGBoost": "xgboost_baseline.pkl",
    "SVM": "svm_baseline.pkl",
    "Decision Tree": "decision_tree_baseline.pkl"
}

all_metrics = {}

best_overall_score = -1
best_overall_name = ""
best_overall_model = None

# BASELINE RESULTS
print("\n\n================ BASELINE RESULTS ================\n")

for name, filename in baseline_files.items():

    model = joblib.load(
        os.path.join(baseline_dir, filename))

    metrics = evaluate(name, model)
    all_metrics[f"{name} (Baseline)"] = metrics

    print(
        f"{name:<25} "
        f"F1: {metrics['F1 Score']:.4f} | "
        f"Recall: {metrics['Recall']:.4f} | "
        f"Precision: {metrics['Precision']:.4f}"
    )

    # Baseline can technically be overall best
    if metrics["F1 Score"] > best_overall_score:

        best_overall_score = metrics["F1 Score"]
        best_overall_name = f"{name} (Baseline)"
        best_overall_model = model

# TUNING
print("\n\n TUNING MODELS ...\n")


for name, (model, params) in tuned.items():

    print("\n" + ">" * 70)
    print(f"Finding best parameters for Tuned {name} ...")

    # GRID SEARCH
    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring="f1",
        cv=tscv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    grid_model = grid.best_estimator_

    grid_metrics = evaluate(
        f"Tuned {name} (Grid)",
        grid_model
    )

    all_metrics[f"Tuned {name} (Grid)"] = grid_metrics

    print("Grid Best: ", grid.best_params_)

    # RANDOM SEARCH
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=min(
            8,
            len(list(ParameterGrid(params)))
        ),
        scoring="f1",
        cv=tscv,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    random_model = random_search.best_estimator_

    random_metrics = evaluate(
        f"Tuned {name} (Random)",
        random_model
    )

    all_metrics[f"Tuned {name} (Random)"] = random_metrics

    print("Random Best:", random_search.best_params_)

    # SELECT BETTER METHOD
    if grid_metrics["F1 Score"] >= random_metrics["F1 Score"]:

        selected_model = grid_model
        selected_metrics = grid_metrics
        selected_method = "Grid Search"
        selected_params = grid.best_params_

    else:

        selected_model = random_model
        selected_metrics = random_metrics
        selected_method = "Random Search"
        selected_params = random_search.best_params_

    print("\nBest Method:", selected_method)
    print("Best Params:", selected_params)

    # save
    clean_name = name.lower().replace(" ", "_")

    joblib.dump(
        selected_model,
        f"models/tuned/{clean_name}.pkl"
    )

    # OVERALL CHAMPION
    if selected_metrics["F1 Score"] > best_overall_score:

        best_overall_score = selected_metrics["F1 Score"]
        best_overall_name = f"{name} ({selected_method})"
        best_overall_model = selected_model

# FINAL COMPARISON

print("\n\n===FINAL COMPARISON===\n")

model_groups = [
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "Decision Tree",
    "Logistic Regression",
    "SVM"
]

for model_name in model_groups:

    print(f"\n### {model_name}")
    print("-" * 75)

    # BASELINE
    baseline_name = f"{model_name} (Baseline)"

    if baseline_name in all_metrics:

        m = all_metrics[baseline_name]

        print(
            f"Baseline      | "
            f"F1: {m['F1 Score']:.4f} | "
            f"Acc: {m['Accuracy']:.4f} | "
            f"Prec: {m['Precision']:.4f} | "
            f"Recall: {m['Recall']:.4f}"
        )

    # GRID SEARCH
    grid_name = f"Tuned {model_name} (Grid)"

    if grid_name in all_metrics:

        m = all_metrics[grid_name]

        print(
            f"Grid Search   | "
            f"F1: {m['F1 Score']:.4f} | "
            f"Acc: {m['Accuracy']:.4f} | "
            f"Prec: {m['Precision']:.4f} | "
            f"Recall: {m['Recall']:.4f}"
        )

    # RANDOM SEARCH
    random_name = f"Tuned {model_name} (Random)"
    if random_name in all_metrics:

        m = all_metrics[random_name]

        print(
            f"Random Search | "
            f"F1: {m['F1 Score']:.4f} | "
            f"Acc: {m['Accuracy']:.4f} | "
            f"Prec: {m['Precision']:.4f} | "
            f"Recall: {m['Recall']:.4f}"
        )

# BEST OVERALL MODEL
print("\n" + "=" * 60)
print("BEST OVERALL MODEL")
print("=" * 60)

print(f"Model: {best_overall_name}")
print(f"F1 Score: {best_overall_score:.4f}")


# SAVE CHAMPION
joblib.dump(
    best_overall_model,
    "models/tuned/Best_Model.pkl"
)

print("\nBest model saved to:")
print("models/tuned/Best_Model.pkl")

print("\n================ TUNING COMPLETE ================\n")