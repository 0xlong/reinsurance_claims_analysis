# 🛡️ Risktec — Motor Insurance Pricing & Underwriting Copilot

A full-stack insurance data engineering and AI project that builds a governed data pipeline and an interactive underwriting tool for motor third-party liability pricing. The system predicts **pure premium** (frequency × severity) for motor policies using XGBoost, explains risk drivers with SHAP, and delivers results through a Streamlit dashboard.

---

[reinsurance_claims_analysis_video.webm](https://github.com/user-attachments/assets/aae3d87b-ffcf-43ca-bdaa-104adf5e2851)

---

## Architecture

```
Raw CSV --> DuckDB --> dbt Staging --> dbt Mart --> XGBoost Models --> Streamlit Dashboard
(Bordereaux)  (Warehouse)  (stg_policies,    (mart_pricing)  (Frequency: Poisson)  (Underwriting Copilot
                            stg_claims,                       (Severity: Gamma)      + Portfolio Overview
                            stg_regions)                      (SHAP Explainability)  + SHAP Risk Explanation)
```

---

## Project Structure

```text
reinsurance_project/
├── data/
│   ├── raw/                          # Raw CSV files (freMTPL2freq, freMTPL2sev, regions)
│   ├── models/                       # Serialized pipelines & SHAP plots
│   │   ├── frequency_pipeline.pkl
│   │   ├── severity_pipeline.pkl
│   │   ├── shap_frequency_summary.png
│   │   └── shap_severity_summary.png
│   └── insurance.db                  # DuckDB local data warehouse
├── dbt_project/
│   ├── models/
│   │   ├── staging/                  # Data cleaning & standardization
│   │   │   ├── stg_policies.sql
│   │   │   ├── stg_claims.sql
│   │   │   ├── stg_regions.sql
│   │   │   ├── schema.yml            # Column docs + governance tests
│   │   │   └── sources.yml
│   │   └── marts/                    # ML-ready analytical tables
│   │       ├── mart_pricing.sql
│   │       └── schema.yml            # Column docs + governance tests
│   ├── dbt_project.yml
│   └── profiles.yml
├── notebooks/
│   └── eda.ipynb                     # Exploratory data analysis
├── app/
│   └── app.py                        # Streamlit underwriting dashboard
├── project_context/
│   ├── project_plan.md               # Original project plan & phases
│   └── project_development_comments.md  # Bug fixes & development notes
├── ingest_data.py                    # Phase 1: Load raw CSVs into DuckDB
├── train_models.py                   # Phase 2: Train frequency & severity models
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+

### 1. Create a Virtual Environment

```bash
python -m venv reinsurance_project_env
source reinsurance_project_env/bin/activate        # macOS/Linux
.\reinsurance_project_env\Scripts\activate          # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Ingest Raw Data into DuckDB

The data has to be downloaded csvs from Kaggle first:



```bash
python ingest_data.py
```

This loads the raw CSV bordereaux files (`freMTPL2freq`, `freMTPL2sev`, `regions`) into DuckDB tables, simulating data intake into a cloud warehouse.

### 4. Run the dbt Data Pipeline

```bash
cd dbt_project
dbt build
cd ..
```

This executes the full governed pipeline:

- **Staging views** (`stg_policies`, `stg_claims`, `stg_regions`): clean types, standardize column names, apply data governance tests (exposure > 0, unique policy IDs, non-negative claim counts/amounts).
- **Mart table** (`mart_pricing`): joins policies with aggregated claims and region metadata to produce the ML-ready analytical table.

### 5. Train the Pricing Models

```bash
python train_models.py
```

This trains two XGBoost models and saves them as serialized scikit-learn pipelines:

- **Frequency model** (`count:poisson`): predicts claim rate per exposure year.
- **Severity model** (`reg:gamma`): predicts average claim cost given a claim occurred.
- SHAP summary plots are generated and saved to `data/models/`.

### 6. Launch the Dashboard

```bash
streamlit run app/app.py
```

The Streamlit app loads the trained pipelines and provides:

- **Portfolio Overview**: historical KPIs, observed pure premium by age bucket, actual vs. expected claims, exposure concentration by region, and actual vs. expected pure premium by region.
- **Underwriting Copilot**: enter policy risk features → get frequency, severity, and pure premium predictions with a SHAP waterfall explanation of the key risk drivers.

---

## Key Technical Decisions

### Why Frequency × Severity (not a single model)?

Insurance pricing separates "how often" (frequency) from "how much" (severity) because they are driven by different risk factors. A young driver in a city may claim more often (high frequency) but have cheaper repairs (low severity). Combining both into a single model would mask these patterns. The **pure premium** is the product: `Pure Premium = Frequency × Severity`.

### Why XGBoost with Poisson / Gamma Objectives?

- **Poisson** is the standard distribution for count data (claim frequency). It naturally handles the fact that most policies have 0 claims.
- **Gamma** is the standard distribution for positive-valued, right-skewed continuous data (claim costs). It handles the long tail of insurance losses.
- XGBoost with these objectives mirrors what actuaries use in Generalized Linear Models (GLMs), but with the flexibility of gradient-boosted trees for capturing non-linear interactions.

### Why SHAP?

Regulators and underwriters need to understand _why_ a premium was set at a certain level. SHAP (SHapley Additive exPlanations) decomposes each prediction into the contribution of each feature, providing transparent, auditable explanations — critical for insurance compliance.

---

## Dataset

The project uses the French motor third-party liability dataset (`freMTPL2`) from OpenML:

- **Frequency data** (`freMTPL2freq`, OpenML ID 41214): 678,013 policies with risk features (driver age, vehicle power, bonus/malus, region, etc.) and claim counts.
  (https://www.kaggle.com/datasets/jesussanchezluengas/fre-mtpl2-dataset?select=freMTPL2freq)
- **Severity data** (`freMTPL2sev`, OpenML ID 41215): individual claim amounts linked to policies by ID.
  (https://www.kaggle.com/datasets/jesussanchezluengas/fre-mtpl2-dataset?select=freMTPL2sev)
- **Regions metadata**: French administrative regions with population data.

---

## Future Enhancements

A production version of this system would add:

| Enhancement                       | Description                                                                                                                                |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **GLM Baselines**                 | Poisson frequency and Gamma severity GLMs for actuarial benchmarking and regulatory comparison against the XGBoost models.                 |
| **Reinsurance Treaty Extraction** | RAG/LLM-powered extraction of treaty clauses (limits, retention, reinstatements) from PDF contracts using Snowflake Cortex or similar.     |
| **Cloud Deployment**              | Migration from local DuckDB to Snowflake with Dagster orchestration for scheduled data refreshes and model retraining.                     |
| **API Serving Layer**             | FastAPI model serving endpoint for integration with policy administration systems and real-time underwriting workflows.                    |
| **CI/CD Pipeline**                | GitHub Actions for automated dbt testing, model validation, and Streamlit deployment on every commit.                                      |
| **Monitoring & Drift**            | Model performance monitoring dashboard tracking prediction drift, actual vs. expected ratios over time, and automated retraining triggers. |
