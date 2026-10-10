# Loan Default Prediction

An end-to-end Machine Learning project that predicts whether a customer may default on a loan. The project covers data extraction from Oracle SQL, exploratory data analysis, preprocessing, model training, model serialization, and serving predictions through a Flask REST API.

## Project Overview

The objective of this project is to build a complete Machine Learning workflow using customer and loan-related features, including income, employment history, loan amount, credit score, previous defaults, and loan type.

The trained model accepts customer details and returns a predicted loan-default class through an HTTP API.

**Project type:** End-to-end supervised Machine Learning
**Problem type:** Binary classification
**Target variable:** `DEFAULT_STATUS`
**Models explored:** Logistic Regression, Random Forest Classifier, XGBoost Classifier
**API framework:** Flask

## Technology Stack

* Python
* pandas and NumPy
* scikit-learn
* XGBoost
* Jupyter Notebook
* joblib
* Flask
* Oracle SQL / Oracle Database
* Git and GitHub

## Project Workflow

1. **Data extraction:** Created and joined four tables in Oracle SQL and exported the query results to CSV.
2. **Exploratory Data Analysis:** Inspected the dataset, checked data types, missing values, duplicate records, and feature distributions.
3. **Data preprocessing:** Used numerical and categorical preprocessing, including imputation, feature scaling, and one-hot encoding where applicable.
4. **Train-test split:** Split the dataset into training and testing sets.
5. **Model training:** Trained and compared Logistic Regression, Random Forest, and XGBoost classifiers.
6. **Model evaluation:** Evaluated model predictions using accuracy, precision, recall, and F1-score.
7. **Model serialization:** Saved the trained preprocessing-and-model pipeline using joblib.
8. **API development:** Built a Flask API that accepts customer information as JSON and returns a prediction.
9. **API testing:** Tested the API locally using PowerShell HTTP requests.

## Project Structure

```text
loan_default_prediction/
├── data/
│   └── raw/
│       └── loan_default_dataset.csv
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── models/
│   ├── preprocessor.joblib
│   └── loan_default_pipeline.joblib
├── src/
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The dataset was created for practice using Oracle SQL. Four tables were joined to generate a consolidated dataset, which was exported to CSV and loaded into Python.

The target column is `DEFAULT_STATUS`:

* `0`: No default
* `1`: Default

Features used by the model include:

* `AGE`
* `GENDER`
* `INCOME`
* `EMPLOYMENT_YEARS`
* `CITY`
* `LOAN_AMOUNT`
* `LOAN_TERM`
* `INTEREST_RATE`
* `LOAN_TYPE`
* `CREDIT_SCORE`
* `PREVIOUS_DEFAULTS`
* `NUMBER_OF_LOANS`

`CUSTOMER_ID` is excluded from the model features because it is an identifier rather than a meaningful predictive feature.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jaydeshmukh411/loan_default_prediction.git
cd loan_default_prediction
```

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Flask API

Run the following command from the project root:

```bash
python app.py
```

The local API will be available at:

`http://127.0.0.1:5000`

### Home endpoint

**Method:** `GET`

**URL:** `/`

Returns a message confirming that the API is running.

### Prediction endpoint

**Method:** `POST`

**URL:** `/predict`

Send a JSON object containing customer features.

Example PowerShell request:

```powershell
$body = @{
    AGE = 35
    GENDER = "Male"
    INCOME = 60000
    EMPLOYMENT_YEARS = 8
    CITY = "Pune"
    LOAN_AMOUNT = 300000
    LOAN_TERM = 36
    INTEREST_RATE = 10.5
    LOAN_TYPE = "Personal"
    CREDIT_SCORE = 700
    PREVIOUS_DEFAULTS = 0
    NUMBER_OF_LOANS = 1
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

### Example response

```json
{
  "prediction": 0,
  "label": "No default predicted",
  "class_probabilities": {
    "0": 0.568,
    "1": 0.432
  }
}
```

The values above illustrate the response format; actual predictions depend on the loaded model and input data.

## Model Evaluation

The classifiers were compared using accuracy, precision, recall, and F1-score on a held-out test set.

The initial dataset contains only 10 rows, with 8 rows used for training and 2 for testing. The resulting metrics are not statistically reliable and should not be used to claim that one model generalizes better than another.

A larger, representative dataset and more rigorous validation are required before drawing conclusions about predictive performance.

## Key Learning Outcomes

* Extracting and combining relational data using Oracle SQL
* Exporting SQL query results to CSV
* Performing exploratory data analysis in pandas
* Building preprocessing pipelines with scikit-learn
* Training and evaluating classification models
* Saving and loading ML pipelines with joblib
* Serving predictions through a Flask REST API
* Testing HTTP endpoints with PowerShell
* Structuring a Machine Learning project for version control

## Limitations

* The dataset is very small and intended for learning.
* Model predictions and probability estimates are not validated for real-world lending.
* The API is intended for local development and has not been configured for production deployment.
* Additional data validation, model monitoring, security, and deployment work would be required for a production system.

## Future Improvements

* Use a larger and more representative dataset.
* Improve validation and model comparison.
* Add robust API input validation and automated tests.
* Add cross-validation and systematic hyperparameter tuning.
* Track model versions and evaluation results.
* Deploy the API to a cloud platform.
* Add continuous integration and deployment.

## Disclaimer

This project is for educational and portfolio purposes only. It must not be used to make actual lending, credit approval, or financial decisions.
