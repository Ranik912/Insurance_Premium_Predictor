# Insurance Premium Predictor

A machine learning application that predicts an insurance premium category from basic user and lifestyle information. The project combines a Random Forest classification pipeline with a FastAPI backend and a Streamlit frontend.

## Project Overview

The application takes the following inputs:

- Age
- Weight
- Height
- Annual income in LPA
- Smoking status
- City
- Occupation

Instead of passing all raw inputs directly to the model, the application engineers additional features that are useful for insurance-risk classification:

- BMI
- Age group
- Lifestyle risk
- City tier

The trained model predicts the corresponding insurance premium category.

## Features

- Machine learning based insurance premium category prediction
- Automated BMI calculation
- Age-group classification
- Lifestyle-risk classification using smoking status and BMI
- City-tier classification
- Categorical feature encoding using One-Hot Encoding
- Random Forest classification
- FastAPI REST API for model inference
- Pydantic input validation
- Streamlit web interface
- Saved trained model using Pickle

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Streamlit
- Requests
- Pickle

## Project Structure

```text
Insurance-Premium-Predictor/
│
├── app.py
├── frontend.py
├── main.py
├── fastapi_ml_model.ipynb
├── insurance.csv
├── patients.json
├── model.pkl
└── README.md
```

### File Description

| File | Description |
|---|---|
| `app.py` | FastAPI application that loads the trained model and exposes the `/predict` endpoint |
| `frontend.py` | Streamlit interface for collecting user information and displaying predictions |
| `fastapi_ml_model.ipynb` | Notebook containing data preprocessing, feature engineering, model training and evaluation |
| `insurance.csv` | Dataset used for training the insurance premium classifier |
| `model.pkl` | Serialized Scikit-learn pipeline containing preprocessing and the Random Forest model |
| `main.py` | Separate FastAPI-based patient management API included in the project |
| `patients.json` | JSON data store used by the patient management API |

## Machine Learning Pipeline

### 1. Data Loading

The training dataset is loaded from `insurance.csv` using Pandas.

### 2. Feature Engineering

The following features are generated from the raw data.

#### BMI

BMI is calculated as:

```text
BMI = weight / height²
```

where height is measured in meters and weight is measured in kilograms.

#### Age Group

Age is converted into four categories:

```text
Age < 25       → young
25–44          → adult
45–59          → middle_aged
60+            → senior
```

#### Lifestyle Risk

Lifestyle risk is determined using smoking status and BMI:

```text
Smoker + BMI > 30  → high
Smoker OR BMI > 27 → medium
Otherwise           → low
```

#### City Tier

Cities are grouped into three tiers:

```text
Tier 1 → Major metropolitan cities
Tier 2 → Listed major/regional cities
Tier 3 → All other cities
```

### 3. Feature Selection

The model uses:

```text
bmi
age_group
lifestyle_risk
city_tier
income_lpa
occupation
```

The original `age`, `weight`, `height`, `smoker`, and `city` fields are transformed into derived features where appropriate.

### 4. Preprocessing

Categorical features are transformed using `OneHotEncoder`.

Categorical features:

```text
age_group
lifestyle_risk
occupation
city_tier
```

Numeric features:

```text
bmi
income_lpa
```

A Scikit-learn `ColumnTransformer` combines both preprocessing paths.

### 5. Model

The classifier is a `RandomForestClassifier` with:

```python
random_state=42
```

The complete preprocessing and classification workflow is stored as a single Scikit-learn `Pipeline`.

### 6. Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)
```

The notebook evaluates the trained model using classification accuracy.

## API

The FastAPI backend is implemented in `app.py`.

### Start the API

Install the required dependencies:

```bash
pip install fastapi uvicorn pandas scikit-learn pydantic
```

Run the server:

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

### Prediction Endpoint

```text
POST /predict
```

Example request:

```json
{
  "age": 30,
  "weight": 65,
  "height": 1.7,
  "income_lpa": 10,
  "smoker": false,
  "city": "Mumbai",
  "occupation": "private_job"
}
```

The backend validates the request using Pydantic, calculates the engineered features, passes them to the trained pipeline, and returns the predicted insurance premium category.

Example response:

```json
{
  "predicted_category": "..."
}
```

The exact category depends on the trained model and input data.

## Streamlit Frontend

The Streamlit application is implemented in `frontend.py`.

Install Streamlit and Requests:

```bash
pip install streamlit requests
```

Run the frontend:

```bash
streamlit run frontend.py
```

The interface allows users to enter their information and submit it to the FastAPI prediction endpoint.

### API Configuration

The frontend currently sends requests to the API configured in:

```python
API_URL = "http://34.226.152.222:8000/predict"
```

For local development, change it to:

```python
API_URL = "http://127.0.0.1:8000/predict"
```

Make sure the FastAPI server is running before using the Streamlit application.

## Running the Complete Application

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd Insurance-Premium-Predictor
```

### Step 2: Install dependencies

```bash
pip install pandas numpy scikit-learn fastapi uvicorn pydantic streamlit requests
```

### Step 3: Start FastAPI

```bash
uvicorn app:app --reload
```

### Step 4: Start Streamlit

Open another terminal and run:

```bash
streamlit run frontend.py
```

### Step 5: Use the application

Open the Streamlit URL shown in the terminal, enter the required information, and click **Predict Premium Category**.

## Model Training

To retrain the model:

1. Open `fastapi_ml_model.ipynb`
2. Place `insurance.csv` in the expected working directory
3. Run the notebook cells sequentially
4. The trained pipeline will be saved as `model.pkl`
5. Restart the FastAPI server so it loads the updated model

## Input Validation

The FastAPI backend validates incoming requests before prediction.

Examples of validation rules:

- Age must be greater than 0 and below 120
- Weight must be greater than 0
- Height must be greater than 0 and below 2.5 meters
- Income must be greater than 0
- Smoking status must be boolean
- Occupation must belong to one of the supported occupation categories

Invalid requests are rejected by FastAPI/Pydantic before reaching the model.

## Design Decisions

### Why Feature Engineering?

Insurance premium categories can depend on combinations of health, lifestyle, demographic and socioeconomic factors. Derived features such as BMI, lifestyle risk and city tier allow the model to capture these relationships more directly.

### Why a Pipeline?

The preprocessing steps and classifier are stored together in a Scikit-learn pipeline. This ensures that the same transformations used during training are automatically applied during inference.

### Why FastAPI?

FastAPI provides a lightweight REST interface around the trained model, making the prediction system easy to integrate with web or other client applications.

### Why Streamlit?

Streamlit provides a simple interactive interface for testing the prediction service without requiring a separate frontend framework.

## Future Improvements

- Add model confidence scores and class probabilities to the API response
- Add comprehensive model evaluation metrics such as precision, recall and F1-score
- Compare Random Forest with XGBoost, Logistic Regression and other classifiers
- Add automated model retraining
- Store predictions and user sessions in a database
- Add input normalization and stronger schema validation
- Improve city classification using a maintained city database
- Containerize the application using Docker
- Deploy the FastAPI backend and Streamlit frontend using a cloud platform
- Add automated tests for API endpoints and feature engineering

## Disclaimer

This project is intended for educational and demonstration purposes. The predicted premium category should not be treated as an actual insurance quote or financial/insurance advice.
