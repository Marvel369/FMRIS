import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import (RandomizedSearchCV, GridSearchCV, TimeSeriesSplit, ParameterGrid )
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import(
    RandomForestClassifier,
    HistGradientBoostingClassifier,
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
    confusion_matrix,
    roc_auc_score,
)
#ensure output directory exists
os.makedirs("models", exist_ok=True)
os.makedirs("models/baseline", exist_ok=True)
os.makedirs(
    "models/tuned",
    exist_ok=True
)

# Load
data = pd.read_csv("data/features.csv")
# print(data.head())
print("\n-------------------------\nDataset shape:",data.shape)

# Train test split + Target
TARGET = "risk_label"

# split feature/Target
X = data.drop(columns = ["Date",TARGET,"peak_price"])
y = data[TARGET].shift(-1)

X=X.iloc[:-1]
y=y.dropna()

# Time Split
split = int(len(X) * 0.75)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTrain:", X_train.shape)
print("Test:", X_test.shape)

# Evaluation

def evaluate(name, model,fit_model=True):

    if fit_model:
        model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:,1] 
    pred = ( prob>= 0.35).astype(int)

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1 Score": round(f1_score(y_test, pred), 4),
        "MCC": round(matthews_corrcoef(y_test, pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, prob), 4)
    }

    return metrics, model

#Models baseline

models = {

    #1 Logistic Regression
    "Logistic Regression":
    Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))

    ]),

    #2 RandomForest
    "Random Forest":
    RandomForestClassifier(
        n_estimators = 200,
        random_state=42
    ),

    #3 Gradient Boosting
    "Gradient Boosting":
    HistGradientBoostingClassifier(
        random_state =42
    ),

    #4 XGBoost
    "XGBoost":
    XGBClassifier(
        n_estimators= 200,
        learning_rate = 0.05,
        max_depth=5,
        random_state=42,
        eval_metric = "logloss"
    ),

    #5 SVM
    "SVM":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model",
             SVC(
                 probability=True,
             ))
        ]),

        #5 Decision Tree
            "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            )
}

#Run baseline
results_list = []
baseline_scores = {}
all_metrics_tracker={}

best_overall_score = -1.0
best_overall_name = ""
best_overall_model_obj = None

for name, model in models.items():
    res, fitted_model = evaluate(
        name,
        model,
        fit_model=True
    )
    results_list.append(res)
    baseline_scores[name] = fitted_model
    all_metrics_tracker[name] =res

results_df = pd.DataFrame(results_list)

# ==============================================
# Best model by F1 Score
baseline_models = results_df.sort_values(by="F1 Score", ascending=False).head(6).reset_index(drop=True)

print("\n============================ BASELINE RESULTS ============================\n\n", baseline_models)
print("\nReason: Highest F1 Score\n")

for index, row in baseline_models.iterrows():
    model_name = row["Model"]
    f1_val = row["F1 Score"]
    print(f"\033[32m- {model_name}")

    clean_name = model_name.lower().replace(" ", "_")
    file_name = f"models/baseline/{clean_name}_baseline.pkl"

    joblib.dump(baseline_scores[model_name], file_name)

print("\n\033[32mAll Baseline Models Saved!!!\033[0m \n-----------------------")

#=============================
# Tuning
#=============================

print("\n\nTuning baseline models ...")

negative = (y_train==0).sum()
positive = (y_train==1).sum()
weight = negative/positive

tscv = TimeSeriesSplit(n_splits=5)

tuned = {
    "Tuned Random Forest": (
        RandomForestClassifier(random_state=42,class_weight = "balanced"),
        {
            "n_estimators":[300,400],
            "max_depth":[3,4],
            "min_samples_leaf":[3,5]
        }
    ),

    "Tuned Gradient Boosting":(
        HistGradientBoostingClassifier(
            random_state = 42,
              class_weight= "balanced"
            ),
            {
                "max_iter":[200,300],
                "learning_rate":[0.03,0.05],
                "max_depth":[3,4]
            }
        ),

    "Tuned XGBoost":(
        XGBClassifier(
        random_state = 42,
        eval_metric = "logloss",
        scale_pos_weight = weight
    ),
        {
            "n_estimators":[200,300],
            "max_depth":[3,5],
            "learning_rate":[0.03,0.05]
        }
    ),

    "Tuned SVM":(
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True,class_weight= "balanced"))
        ]),
        {
            "model__C":[0.1, 1.0],
            "model__gamma": ["scale","auto"]
        }
    ),

    "Tuned Decision Tree":(
        DecisionTreeClassifier(random_state=42, class_weight= "balanced"),
            {
                "max_depth":[3,5],
                "min_samples_leaf":[3,5]
            }
    ),

    "Tuned Logistic Regression":(
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        {
            "model__C": [4, 5, 6]
        }
    )
    
}

for name, (model, params) in tuned.items():

    print(f"\n\n{'='*60}")
    print(f"Finding best parameters for {name} ...")

# ------------- Grid search ------------

    grid = GridSearchCV(
        model,
        params,
        scoring='f1',
        cv=tscv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    g_metrics, g_best_model = evaluate(f"{name} (Grid)", grid.best_estimator_, fit_model=False)
    all_metrics_tracker[f"{name} (Grid)"] = g_metrics

    print("Grid Best: ", grid.best_params_)

    # Track if Grid Result wins overall
    if g_metrics["F1 Score"] > best_overall_score:
        best_overall_score = g_metrics["F1 Score"]
        best_overall_name = f"{name} (Grid)"
        best_overall_model_obj = g_best_model

# ---------------- Random Search ----------------

    random_search = RandomizedSearchCV(
        model,
        params,
        n_iter=min(8, len(list(ParameterGrid(params)))),
        scoring='f1',
        cv=tscv,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    random_metrics, random_model = evaluate(
        f"{name} (Random)",
        random_search.best_estimator_,
        fit_model=False
    )

    all_metrics_tracker[f"{name} (Random)"] = random_metrics
    print("Random Best:", random_search.best_params_,"\n")

    # Track if random best
    if random_metrics["F1 Score"] > best_overall_score:
        best_overall_score = random_metrics["F1 Score"]
        best_overall_name = f"{name} (Random)"
        best_overall_model_obj = random_model

    if g_metrics["F1 Score"] >= random_metrics["F1 Score"]:
        print("Best Method: Grid Search")
        print("Best Params:", grid.best_params_)
        clean_name = name.lower().replace(" ", "_")
        joblib.dump(g_best_model, f"models/tuned/{clean_name}.pkl")
    else:
        print("\nBest Method: Random Search")
        print("Best Params:", random_search.best_params_)
        clean_name = name.lower().replace(" ", "_")
        joblib.dump(random_model, f"models/tuned/{clean_name}.pkl")

# Final Ranking
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

    # Baseline
    if model_name in all_metrics_tracker:
        m = all_metrics_tracker[model_name]
        print(
            f"Baseline      | "
            f"F1: {m['F1 Score']:.4f} | "
            f"Acc: {m['Accuracy']:.4f} | "
            f"Prec: {m['Precision']:.4f} | "
            f"Recall: {m['Recall']:.4f} | "
            f"MCC: {m['MCC']:.4f}"
        )

    # Grid Search
    grid_name = f"Tuned {model_name} (Grid)"

    if grid_name in all_metrics_tracker:
        m = all_metrics_tracker[grid_name]
        print(
            f"Grid Search   | "
            f"F1: {m['F1 Score']:.4f} | "
            f"Acc: {m['Accuracy']:.4f} | "
            f"Prec: {m['Precision']:.4f} | "
            f"Recall: {m['Recall']:.4f} | "
            f"MCC: {m['MCC']:.4f}"
        )

    # Random Search
    random_name = f"Tuned {model_name} (Random)"

    if random_name in all_metrics_tracker:
        m = all_metrics_tracker[random_name]
        print(
            f"Random Search | "
            f"F1: {m['F1 Score']:.4f} | "
            f"Acc: {m['Accuracy']:.4f} | "
            f"Prec: {m['Precision']:.4f} | "
            f"Recall: {m['Recall']:.4f} | "
            f"MCC: {m['MCC']:.4f}"
        )

# EXHAUSTIVE OVERALL CHAMPION SUMMARY PRINT OUT
print("\n" + "=" * 40)
print("BEST OVERALL TUNED MODEL")
print("=" * 40,'\n')
print(f"\033[32m{best_overall_name}\033[0m")
print(f"BEST F1-Score: {best_overall_score:.4f}")
print("=" * 60 + "\n")

# Save a dedicated file for the best model framework
joblib.dump(best_overall_model_obj, "models/tuned/Best_Model.pkl")
print(
    "\033[32mBest Model successfully cached to: models/tuned/production_champion.pkl\033[0m\n"
)