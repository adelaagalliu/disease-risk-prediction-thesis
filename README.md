# Health Risk Assessment Platform

A full-stack application for estimating the risk of four health outcomes:

- diabetes or prediabetes;
- heart disease;
- hypertension;
- stroke.

The project combines logistic regression models, an authenticated API, PostgreSQL history storage, and a React interface. Results include explanatory factors, general health recommendations, and an optional AI-generated summary when configured.

> [!IMPORTANT]
> This project was developed for academic and demonstration purposes. Its outputs are statistical estimates, not diagnoses or medical advice, and must not replace an assessment by a qualified healthcare professional.
![Health risk assessment dashboard](docs/images/health_dashboard.png)
![Health risk assessment dashboard](docs/images/health_data.png)
![Health risk assessment dashboard](docs/images/health_shap_scenarios.png)
## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Evaluation](#model-evaluation)
- [API](#api)
- [Testing](#testing)
- [Security and Privacy](#security-and-privacy)
- [Limitations](#limitations)
- [Maintaining Results](#maintaining-results)

## Key Features

- User registration and login with JWT authentication
- A health questionnaire with field validation
- Predictions for four health outcomes
- Classification thresholds selected on the validation set
- Global and individual explanations with SHAP
- General recommendations from a curated catalog
- Optional AI-generated summaries with a fallback when the service is unavailable
- User profile and result storage in PostgreSQL
- Append-only prediction history and health metric trend charts
- User-level data isolation
- Descriptive analyses and inferential logistic regression

## Architecture

```mermaid
flowchart LR
    U[User] --> F[React + Vite]
    F -->|REST / JSON| A[FastAPI]
    A --> M[Logistic Regression Models]
    A --> G[Explanations and Guidance]
    A --> D[(PostgreSQL)]
    G -. optional .-> O[OpenAI API]
    M --> R[Probability, Class, and Factors]
    R --> A
    A --> F
```

The frontend communicates with the API through the `/api` prefix. During development, Vite proxies this prefix to `http://127.0.0.1:8001`. `compose.yaml` starts PostgreSQL only; the backend and frontend run separately.

## Technology Stack

| Layer | Main Technologies |
|---|---|
| Frontend | React 19, Vite 7 |
| Backend | FastAPI, Pydantic, Uvicorn |
| Machine learning | pandas, scikit-learn, SciPy, joblib |
| Explainability | SHAP |
| Database | PostgreSQL 17, SQLAlchemy, Alembic |
| Security | JWT, Argon2 |
| Testing | Node.js test runner, pytest |
| Local infrastructure | Docker Compose for PostgreSQL |

## Project Structure

```text
diploma-uni/
├── backend/
│   ├── app/                    # API, authentication, and services
│   ├── alembic/                # Database migrations
│   └── tests/                  # Backend tests
├── contracts/                  # Prediction contracts
├── data/
│   ├── datasets_original/      # Source datasets
│   ├── datasets_cleaned/       # Cleaned datasets
│   └── datasets_processed/     # Train/validation/test splits
├── frontend/
│   ├── src/                    # React interface
│   └── tests/                  # Frontend integration tests
├── models/                     # Models and selection configuration
├── notebooks/                  # Exploratory analysis and cleaning
├── results/
│   ├── calibration_plots/      # Calibration plots
│   ├── shap/                   # SHAP reports
│   └── statistical_analysis/   # Inferential analyses
├── src/                        # Machine learning pipeline
├── .env.example
├── compose.yaml
└── pyproject.toml
```

## Installation and Setup

### Prerequisites

- Python 3.11 or newer
- Node.js 20.19 or newer
- pnpm through Corepack
- Docker Desktop or an existing PostgreSQL installation

### 1. Configure environment variables

From the project root, create a `.env` file based on `.env.example`:

```powershell
Copy-Item .env.example .env
```

Example local configuration:

```dotenv
DATABASE_URL=postgresql+psycopg://diploma_app:replace_me@localhost:5433/disease_risk
JWT_SECRET_KEY=replace_with_a_long_random_secret
ACCESS_TOKEN_MINUTES=60

POSTGRES_DB=disease_risk
POSTGRES_USER=diploma_app
POSTGRES_PASSWORD=replace_me
POSTGRES_PORT=5433

# Optional
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
```

`POSTGRES_PASSWORD`, the port, and the credentials in `DATABASE_URL` must match. Use a long, random value for `JWT_SECRET_KEY`. Do not commit real credentials to Git.

### 2. Start PostgreSQL

```powershell
docker compose up -d postgres
docker compose ps
```

To stop the service:

```powershell
docker compose down
```

The `postgres_data` volume retains data after the container stops. `docker compose down -v` deletes the data and should only be used when a full reset is intended.

### 3. Install and start the backend

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

Apply database migrations and start the API:

```powershell
Set-Location backend
python -m alembic upgrade head
python -m uvicorn app.main:app --reload --port 8001 --env-file ..\.env
```

Useful endpoints:

- Health check: `http://127.0.0.1:8001/health`
- Swagger documentation: `http://127.0.0.1:8001/docs`
- ReDoc documentation: `http://127.0.0.1:8001/redoc`

The API loads the models at startup, so the `.joblib` files must be available in `models/`.

### 4. Install and start the frontend

In a second terminal:

```powershell
Set-Location frontend
Corepack enable
pnpm install
Copy-Item .env.example .env
pnpm dev
```

Open `http://localhost:5173`. The default `VITE_API_URL=/api` setting uses Vite's proxy to reach the API on port 8001.

### Without Docker

Docker is optional. You can use an existing PostgreSQL instance as long as `DATABASE_URL` points to the correct database and the user has permission to run migrations.

## Machine Learning Pipeline

The pipeline uses logistic regression only. From the `src` directory, run:

```powershell
Set-Location src
python split_data.py
python train.py
python evaluate.py
python statistical_analysis.py
python generate_shap_reports.py
```

Run the steps in this order:

1. `split_data.py` creates the train, validation, and test splits while keeping repeated profile groups in the same split.
2. `train.py` trains the pipelines and saves the artifacts in `models/`.
3. `evaluate.py` selects a threshold on the validation set and applies it unchanged to the test set.
4. `statistical_analysis.py` creates descriptive statistics and inferential tables using only the training set.
5. `generate_shap_reports.py` generates global importance reports and individual SHAP examples.

Cleaned data must already be available in `data/datasets_cleaned/`.

## Model Evaluation

The models are compared with a baseline that returns a constant probability equal to the prevalence in the training set. Metrics include ROC-AUC, PR-AUC, F1, sensitivity, specificity, Brier score, and calibration. Confidence intervals are calculated using group-level bootstrapping.

Test-set results using thresholds selected on the validation set:

| Outcome | Threshold | ROC-AUC | PR-AUC | F1 | Sensitivity | Specificity | Brier score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diabetes/prediabetes | 0.21 | 0.822 | 0.449 | 0.488 | 0.679 | 0.793 | 0.107 |
| Heart disease | 0.17 | 0.846 | 0.373 | 0.413 | 0.606 | 0.862 | 0.070 |
| Hypertension | 0.34 | 0.946 | 0.885 | 0.817 | 0.856 | 0.892 | 0.087 |
| Stroke | 0.19 | 0.804 | 0.196 | 0.222 | 0.220 | 0.953 | 0.052 |

Interpret PR-AUC and F1 alongside outcome prevalence, especially for heart disease and stroke. Interpret the hypertension result with caution: the dataset documentation does not fully explain how the `Risk` label was created, so overlap with the blood pressure measurements used as predictors cannot be ruled out.

## API

All prediction and history endpoints require a JWT, except for registration, login, and the `/health` check.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Checks API health |
| `POST` | `/auth/register` | Creates a user |
| `POST` | `/auth/login` | Returns a JWT |
| `GET` | `/users/me` | Returns the authenticated user's profile |
| `POST` | `/predict/{disease}` | Runs and saves a prediction |
| `GET` | `/history/predictions` | Returns the user's prediction list |
| `GET` | `/history/predictions/{prediction_id}` | Returns details for one result |
| `GET` | `/history/trends/{metric}` | Returns a time series for a metric |

Valid values for `{disease}` are `diabetes`, `heart_disease`, `hypertension`, and `stroke`. History metrics include `bmi`, `weight_kg`, `systolic_bp`, `diastolic_bp`, `heart_rate`, `glucose`, and `cholesterol`.

Typical request flow:

1. Register with `POST /auth/register`.
2. Get a token with `POST /auth/login`.
3. Send `Authorization: Bearer <access_token>`.
4. Call `POST /predict/{disease}`.
5. Use the `/history` endpoints to retrieve saved results and trends.

## Testing

The project includes 22 listed tests: 4 frontend integration tests and 18 backend tests.

### Frontend

```powershell
Set-Location frontend
pnpm test
```

The tests cover questionnaire validation, payload construction, authentication, partial failures, and loading history without running a new prediction.

### Backend

With the virtual environment activated:

```powershell
Set-Location backend
python -m pytest -q
```

The tests cover API health, authentication and history, model contracts, health guidance, and recommendations for all four outcomes.

## Security and Privacy

- Passwords are stored as Argon2 hashes, not as plain text.
- Access tokens have a configurable lifetime.
- History requests are scoped to the authenticated user's UUID.
- The input profile and result are saved in the same transaction.
- Prediction history is append-only; older measurements are not overwritten.
- Deleting a user cascades to their health records and predictions.
- Without `OPENAI_API_KEY`, the application continues to work using approved fallback content.

A production deployment would also require HTTPS, centralized secret management, rate limiting, audit logs, backups, and documented data retention policies.

## Limitations

- The models were trained on retrospective datasets and require external and prospective validation.
- Inferential associations do not establish causation.
- Class imbalance particularly limits performance for stroke.
- The origin and construction criteria for the hypertension dataset's `Risk` label are not fully documented.
- Blood pressure measurements may overlap with how the hypertension label was created, so the high performance should not automatically be interpreted as prospective prediction.
- Some inferential estimates, such as the `bp_meds` coefficient, are unstable because of near-complete separation and should not be interpreted clinically.
- Passing technical tests does not certify the application for clinical use.

## Maintaining Results

When the data, features, models, or thresholds change:

1. rerun the full pipeline;
2. update the artifacts in `models/`;
3. verify the contracts in `contracts/`;
4. run all 22 tests;
5. review the metrics, calibration, and SHAP reports;
6. document the evaluation date or commit.
