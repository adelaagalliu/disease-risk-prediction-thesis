# Disease risk: Logistic Regression and statistical baselines

The project uses Logistic Regression as its only predictive model. Run the
pipeline from `src` in this order:

```powershell
python split_data.py
python train.py
python evaluate.py
python statistical_analysis.py
python generate_shap_reports.py
```

`evaluate.py` compares Logistic Regression with a constant training-prevalence
baseline using ROC-AUC, PR-AUC, F1, Brier score, calibration, sensitivity, and
specificity. The classification threshold is selected on validation data and
then applied unchanged to the test data.

`statistical_analysis.py` uses the training split only and writes descriptive
statistics, classical group tests, and univariable/multivariable Logistic
Regression tables (odds ratios, 95% confidence intervals, and p-values) to
`results/statistical_analysis`.

## PostgreSQL, users, and prediction history

The API stores three related types of data:

- `users`: normalized email addresses, Argon2 password hashes, account status,
  and audit timestamps. Plain-text passwords are never stored.
- `health_records`: immutable snapshots of the validated input used for a
  prediction. Frequently graphed numeric values are relational columns; the
  complete request is also retained as JSONB.
- `predictions`: disease probability, classification, threshold, model name and
  version, complete response details, and creation time.

Each query is restricted to the authenticated user's UUID. Deleting a user
cascades to that user's health records and predictions.

### Local setup

1. Copy `.env.example` to `.env`, replace both placeholder secrets, and load
   those variables in the shell that runs the API.
2. Start PostgreSQL with `docker compose up -d postgres` from the project root.
3. Install dependencies with `pip install -r backend/requirements.txt`.
4. From `backend`, run `alembic upgrade head`.
5. Start the API with `uvicorn app.main:app --reload`.

Generate `JWT_SECRET_KEY` with a cryptographically secure random generator; do
not use the development fallback outside a local machine.

### API flow

1. `POST /auth/register` with `email` and a password of at least 12 characters.
2. `POST /auth/login`, then send `Authorization: Bearer <access_token>`.
3. Call `POST /predict/{disease}`. The response includes `prediction_id` and
   `created_at`, and the input/result are committed in one database transaction.
4. Use `GET /history/predictions?disease=diabetes` for disease-risk charts.
5. Use `GET /history/trends/{metric}` for `bmi`, `weight_kg`, `systolic_bp`,
   `diastolic_bp`, `heart_rate`, `glucose`, or `cholesterol` charts.

The history is intentionally append-only. A new measurement or prediction adds
a dated row instead of overwriting an older one.
