import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import(
    RandomForestClassifier,
    GradientBoostingClassifier,
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

def evaluate(name, model):

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:,1] 
    pred = ( prob>= 0.35).astype(int)

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1 Score": round(f1_score(y_test, pred), 4),
        "MCC": round(matthews_corrcoef(y_test, pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, prob), 4)
    }

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
    GradientBoostingClassifier(
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

for name, model in models.items():
    res=evaluate(
        name,
        model
    )
    results_list.append(res)

results_df = pd.DataFrame(results_list)

print("\n============================BASELINE============================\n")
print(
    results_df.sort_values(
        by="F1 Score",
    ).reset_index(drop=True)
)
# ========================================================================

# Best model by F1 Score
top_3_df = results_df.sort_values(by="F1 Score", ascending=False).head(3).reset_index(drop=True)

print("\n======Top 3 Models======\n", top_3_df)
print("\nReason: Highest F1 Score\n")

for index, row in top_3_df.iterrows():
    model_name = row["Model"]
    f1_score = row["F1 Score"]
    print(f"\033[32m- {model_name}")

    file_name = f"models/{model_name.lower()}_baseline.pkl"
    joblib.dump(models[model_name], file_name)

print("\n\033[32mAll Baseline Models Successfully Saved for fine-tuning!!!\033[0m \n-----------------------")

