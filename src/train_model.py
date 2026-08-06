import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import(
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    mean_absolute_error,
)

# load

data = pd.read_csv("data/features.csv")
print(data.head())
print("dataset shape:",data.shape)

# train test split + target

TARGET = "risk_label"

# split feature/target
X = data.drop(columns = ["Date",TARGET,"drawdown","market_return","vix"],axis=1)
y = data[TARGET]

#train test split
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

#Models

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
                 random_state=42
             ))
        ])
}

#train
results = {}
for name, model in models.items():
    print(f"\n{name}")
    print("-" *40)

    #train
    model.fit(X_train, y_train)

    #Prediction
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] # grabs every row and only the 1 index(which is prob of crash) and not 0 prob of normal.

    # metrics (y_test_answer, y_prediction)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save results
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")

    print("\n=== Confusion Matrix ===")
    print(cm)

    #save model
    filename = name.lower().replace(" ","-") + ".pkl"
    joblib.dump(model, f"models/{filename}")





