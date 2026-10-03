# Loan Status Prediction — Deployment Version

## Files

- `train_model_deploy.py` — trains the original Naïve Bayes + KNN(k=3) setup and saves extra evaluation metadata.
- `app_deploy.py` — Streamlit application with model comparison, confusion matrix, per-class metrics, probability explanation, and downloadable prediction details.
- `requirements_deploy.txt` — pinned ML versions to avoid the saved-model version mismatch.
- `loan_data (1) (1).csv` — dataset.

## Local setup

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements_deploy.txt
```

Train:

```bash
python train_model_deploy.py "loan_data (1) (1).csv"
```

Run:

```bash
streamlit run app_deploy.py
```

## Deployment

Deploy `app_deploy.py`, `loan_model_deploy.joblib`, and `requirements_deploy.txt` together with the dataset only if you want the dataset available for project inspection. The app itself does not need the CSV after the model bundle has been created.

For a hosted Streamlit deployment, set the main file to `app_deploy.py` and use `requirements_deploy.txt` as the dependency file (or rename it to `requirements.txt` in the deployment repository).

## Important

This is an educational ML demonstration. The dataset has class imbalance and limited separation between the classes, so the model should not be used as the sole basis for real credit decisions.
