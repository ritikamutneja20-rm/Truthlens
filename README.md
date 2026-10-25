# TruthLens — Fake News Detection System

TruthLens is a Flask-based web application that uses Machine Learning to classify news articles as **Likely Real** or **Likely Fake**. It includes user registration, login, a protected dashboard, and news prediction.

> **Disclaimer:** Predictions are estimates made by a machine-learning model. They do not verify whether a news article is factually true.

## Features

- User registration and login
- SQLite database for user accounts
- Protected dashboard and logout
- News text classification
- TF-IDF text feature extraction
- Logistic Regression classifier
- Model training and evaluation scripts

## Tech Stack

- Python
- Flask
- SQLite
- Pandas
- Scikit-learn
- Joblib
- HTML and CSS

## Project Structure

```text
fake_news_detector_fixed/
├── app.py
├── train_model.py
├── test_model.py
├── model.joblib
├── Fake.csv
├── True.csv
├── requirements.txt
├── README.md
├── .gitignore
├── templates/
└── static/
```

The `templates/` and `static/` folders contain the website's frontend files. Your actual project may include additional files.

## Requirements

- Python 3.10 or later, compatible with the installed dependencies
- pip
- VS Code (recommended)

## Setup and Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

Replace the placeholders with your actual repository URL and folder name.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
```

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Configure the secret key

```powershell
$env:SECRET_KEY = "replace-with-a-unique-random-secret"
```

Use a strong, private key for your own local session. Never commit secrets to GitHub.

### 5. Run the application

```powershell
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000 in your browser.

The SQLite database is created by the application if required.

## Train the Model

Ensure `Fake.csv` and `True.csv` are available in the project directory, with the expected text columns.

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The training script combines titles and article text, splits the dataset into training and testing portions, trains a TF-IDF + Logistic Regression Pipeline, evaluates it, and saves `model.joblib`.

## Test the Model

```powershell
.\.venv\Scripts\python.exe test_model.py
```

This script evaluates predictions on a sample of the available dataset. Since these articles may overlap with the data used during training, its score should not be treated as an independent estimate of real-world performance.

## Model Evaluation

The current random train/test split reported approximately **99.30% accuracy**. This result applies to the dataset split and does not guarantee similar performance on new or current news articles. Similar or syndicated articles may also inflate evaluation scores.

## Security Notes

- Do not commit `.env`, private keys, `users.db`, or other sensitive information.
- Keep `.venv/` and Python cache files out of version control.
- Check dataset licensing and size before publishing the datasets.
- Before public deployment, disable debug mode and configure secure hosting, HTTPS, CSRF protection, and rate limiting.

## Limitations

TruthLens is a text classifier, not a fact-checking service. It does not independently verify sources, dates, evidence, or claims. Always check important news against reliable sources.

## Future Improvements

- Evaluate on independent, newer news articles
- Improve generalization and reduce dataset bias
- Add source verification and explainable predictions
- Improve input validation and security

## Author

Add your name and relevant project details here.