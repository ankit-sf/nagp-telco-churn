DEMO Link: 
Github link: https://github.com/ankit-sf/nagp-telco-churn


# Customer Churn Prediction

This project predicts customer churn for a telecommunications company using the IBM Telco Customer Churn dataset.

The final solution includes:

- Data understanding and preparation
- Exploratory data analysis
- Feature engineering
- Decision Tree model development
- Model comparison
- Model evaluation
- Feature importance interpretation
- Saved preprocessing/model pipeline
- FastAPI prediction API

## Project Structure

```text
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── features/
│   ├── __init__.py
│   └── build_features.py
├── notebook/
│   └── churn_analysis.py
├── model/
│   └── churn_model.pkl
├── outputs/
│   ├── model_comparison.csv
│   ├── final_model_metrics.json
│   ├── feature_importance.csv
│   └── confusion_matrix.png
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── sample_request.json
```

## Dataset

Place the following files inside the `data/` folder:

```text
data/TelcoCustomerChurn.csv
data/TelcoCustomerChurn - Data Dictionary.csv
```

Target variable:

```text
Churn
```

Values:

```text
Yes / No
```

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Model Training

```bash
python train_model.py
```

This trains two Decision Tree configurations:

1. Decision Tree with `max_depth=4`
2. Decision Tree with `max_depth=6` and `class_weight="balanced"`

The script saves:

```text
model/churn_model.pkl
model/model_metadata.json
outputs/model_comparison.csv
outputs/final_model_metrics.json
outputs/feature_importance.csv
outputs/confusion_matrix.png
```

## Notebook

The notebook-style analysis is available in:

```text
notebook/churn_analysis.py
```

Open this file in VS Code or Jupyter as notebook cells. You can export it as:

```text
notebook/churn_analysis.ipynb
```

## Feature Engineering

Feature engineering code is kept in:

```text
features/build_features.py
```

Created features include:

### 1. `tenure_group`

Groups customers by tenure stage:

- `0-12 months`
- `13-24 months`
- `25-48 months`
- `49-72 months`

This is useful because new customers and long-term customers usually have different churn behaviour.

### 2. `avg_charge_per_tenure_month`

Calculated as:

```text
TotalCharges / tenure
```

This gives a rough customer value or billing intensity indicator.

### 3. `is_month_to_month`

Flags whether the customer has a month-to-month contract.

### 4. `has_security_and_support`

Flags whether the customer has both online security and tech support.

### 5. `has_streaming_services`

Flags whether the customer uses streaming TV or streaming movies.

## API

Start the API:

```bash
uvicorn app:app --reload
```

API will run at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Prediction Endpoint

### POST `/predict`

Example request:

```json
{
  "customer": {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": "29.85"
  }
}
```

Example response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.82
}
```

## Test API with curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Business Interpretation

For telecom churn prediction, recall is usually more important than precision.

Reason:

- A false negative means the company misses a customer who is actually likely to churn.
- A false positive usually means the company contacts a customer who may not have churned anyway.
- In retention campaigns, missing true churners can be more costly than contacting extra customers.

Therefore, the model selection considers F1 score and recall strongly.

## Important Preprocessing Decisions

- `customerID` is removed because it is an identifier.
- `TotalCharges` is converted to numeric because it may contain blank strings.
- Missing numerical values are imputed using the median.
- Missing categorical values are imputed using the most frequent value.
- Categorical features are encoded using One-Hot Encoding.
- `handle_unknown="ignore"` is used so the API can safely handle unseen categories.
- Feature engineering and preprocessing are saved inside one sklearn pipeline to avoid training-serving mismatch.
- Train/test split uses 70:30 with `random_state=42`.
