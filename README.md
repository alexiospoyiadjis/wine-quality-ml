# Wine Quality Prediction — ML Pipeline Comparison & Model Serving

## Overview

This project compares two machine learning workflows — **PyCaret** (low-code) and **scikit-learn** (manual) — for predicting wine quality scores using the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality). The best model is then deployed as a REST API using **FastAPI**.

The dataset combines red and white Portuguese "Vinho Verde" wine samples (6,497 total rows) with 11 physicochemical features and a categorical quality target (scores 3–9).

## Project Structure

```
├── discovery.py              # PyCaret vs sklearn comparison script
├── main.py                   # FastAPI model serving application
├── data/
│   ├── winequality-red.csv   # Red wine samples (1,599 rows)
│   └── winequality-white.csv # White wine samples (4,898 rows)
├── best_pipeline.pkl         # Saved PyCaret model pipeline (generated)
├── report.docx               # Screenshots and evaluation report
└── README.md
```

## Setup & Installation

```bash
pip install pycaret fastapi uvicorn pandas scikit-learn
```

## Usage

### Step 1: Run the Discovery Script

```bash
python discovery.py
```

This will:
- Load and combine the red/white wine datasets
- Run PyCaret's `compare_models()` to identify top classifiers
- Generate a confusion matrix for the best model
- Manually implement the same model in scikit-learn with a classification report
- Save the best pipeline as `best_pipeline.pkl`

### Step 2: Start the API

```bash
uvicorn main:app --reload
```

Then visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the Swagger UI.

### Step 3: Test a Prediction

**Sample Input (POST to `/predict`):**

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4,
  "type": "red"
}
```

**Sample Output:**

```json
{
  "prediction": "5",
  "label": "Predicted wine quality: 5/9"
}
```

![Swagger UI Screenshot](swagger_screenshot.png)

## Key Findings

- **PyCaret** completed model comparison across 15+ classifiers in ~3 lines of code, automatically handling preprocessing, cross-validation, and ranking.
- **Scikit-learn** required explicit implementation of each pipeline stage (encoding, scaling, splitting, training, evaluation) but offered full transparency and control.
- Both workflows converged on the same top-performing model, validating reliability.
- Minor accuracy differences stem from PyCaret's stratified k-fold CV vs. sklearn's single train/test split, plus default preprocessing differences.

## Dataset Source

P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis.
*Modeling wine preferences by data mining from physicochemical properties.*
Decision Support Systems, 47(4):547–553, 2009.
[UCI ML Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)
