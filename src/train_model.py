import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
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
def train_models():
    data = pd.read_csv("data/features.csv")
    print(data.head())
    print("dataset shape:",data.shape)

# train test split + target

TARGET = "risk_label"










if __name__ == "__main__":
    train_models()
