
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading datasets...")

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

for name, df in [("Fake.csv", fake), ("True.csv", real)]:
    if "text" not in df.columns:
        raise ValueError(f"{name} must contain a 'text' column.")

# Combine title and text
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

print("Total articles:", len(data))
print("Fake articles:", int((data["label"] == 0).sum()))
print("Real articles:", int((data["label"] == 1).sum()))

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

print("\nTraining model...")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["Fake", "Real"],
    zero_division=0
))

print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nChecking held-out samples:")
for expected, text in [
    ("REAL", X_test[y_test == 1].iloc[0]),
    ("FAKE", X_test[y_test == 0].iloc[0])
]:
    prediction = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    print(f"Expected: {expected}; Predicted: {'REAL' if prediction == 1 else 'FAKE'}")
    print(f"Probabilities [FAKE, REAL]: {probabilities}")

joblib.dump(model, "model.joblib")
print("\nModel saved successfully as model.joblib")
