
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent

fake_path = BASE_DIR / "Fake.csv"
true_path = BASE_DIR / "True.csv"
model_path = BASE_DIR / "model.joblib"

# Check required files
for path in [fake_path, true_path, model_path]:
    if not path.exists():
        raise FileNotFoundError(f"File missing: {path.name}")

print("Loading datasets and model...")

fake = pd.read_csv(fake_path)
real = pd.read_csv(true_path)
model = joblib.load(model_path)

# Validate required columns
for name, df in [("Fake.csv", fake), ("True.csv", real)]:
    if "text" not in df.columns:
        raise ValueError(f"{name} must contain a 'text' column.")

# Match the preprocessing used in train_model.py
fake["title"] = fake.get("title", "").fillna("").astype(str)
real["title"] = real.get("title", "").fillna("").astype(str)

fake["text"] = fake["title"] + " " + fake["text"].fillna("").astype(str)
real["text"] = real["title"] + " " + real["text"].fillna("").astype(str)

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real], ignore_index=True)
data = data.dropna(subset=["text"])
data["text"] = data["text"].astype(str).str.strip()
data = data[data["text"] != ""]
data = data.drop_duplicates(subset=["text"])

# Reproduce the same held-out test split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# Evaluate only on the held-out test portion
y_pred = model.predict(X_test)

print("\n--- TruthLens Held-Out Model Evaluation ---")
print(f"Training articles: {len(X_train)}")
print(f"Testing articles: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Fake", "Real"],
    zero_division=0
))

print("Confusion Matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

