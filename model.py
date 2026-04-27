# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import joblib

# Load
df = pd.read_csv("AI Job Market Dataset.csv")

# Drop unnecessary
df = df.drop(columns=["job_id"])

# Simplify roles
def simplify_role(title):
    title = title.lower()
    if "data" in title:
        return "Data"
    elif "engineer" in title:
        return "Engineer"
    elif "analyst" in title:
        return "Analyst"
    else:
        return "Other"

df["job_title"] = df["job_title"].apply(simplify_role)

# Target
y = df["job_title"]
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Features
X = df.drop(columns=["job_title"])
X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ensemble model
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6)
rf = RandomForestClassifier(n_estimators=200)

model = VotingClassifier(
    estimators=[("xgb", xgb), ("rf", rf)],
    voting="soft"
)

model.fit(X_train, y_train)

# Save everything
joblib.dump(model, "job_model.pkl")
joblib.dump(le_target, "target_encoder.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("✅ Model ready!")