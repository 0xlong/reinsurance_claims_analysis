# Risktec Data & AI Engineer Portfolio Project Plan

## 1. Business Problem & "Why risktec Cares"

A reinsurer needs to price motor third-party liability risk accurately to optimize their treaty portfolio. This project builds a governed data pipeline and an AI-powered pricing tool that predicts **pure premium** (frequency × severity) for motor policies — demonstrating an **insurance solution that happens to use ML**, not just an ML exercise.

## 2. Instructions for the AI / LLM

1. **Act as a Senior Re/insurance Data & AI Engineer**: Write clean, modular, well-documented code.
2. **Use Domain Language Everywhere**: Use _exposure_ (not sample weight), _pure premium_ (not predicted value), _loss ratio_, _bordereaux_ (for data ingestion), _treaty/facultative_ (for reinsurance context). Frame dbt tests as "data governance controls."
3. **Keep It Simple**: Prefer clarity over cleverness. No unnecessary abstractions. The goal is a clean, explainable project.
4. **Work Step-by-Step**: Complete each phase sequentially, verifying outputs before moving to the next.

## 3. Target Technology Stack

- **Data Pipeline**: Python, DuckDB (local data warehouse), dbt (transformations + data governance).
- **Machine Learning**: XGBoost, Scikit-Learn, SHAP (interpretability).
- **Dashboard**: Streamlit (interactive underwriting tool — calls models directly, no separate API needed).

## 4. Project Scaffolding

### Data Context: Description

In the dataset `freMTPL2freq` risk features and claim numbers were collected for 677,991 motor third-part liability policies (observed on a year).

`freMTPL2freq` contains 11 columns (+IDpol) :

- IDpol: The policy ID (used to link with the claims dataset).
- ClaimNb: Number of claims during the exposure period.
- Exposure: The exposure period.
- Area: The area code.
- VehPower: The power of the car (ordered categorical).
- VehAge: The vehicle age, in years.
- DrivAge: The driver age, in years (in France, people can drive a car at 18).
- BonusMalus: Bonus/malus, between 50 and 350: <100 means bonus, >100 means malus in France.
- VehBrand: The car brand (unknown categories).
- VehGas: The car gas, Diesel or regular.
- Density: The density of inhabitants (number of inhabitants per km2) in the city the driver of the car lives in.
- Region: regions in France (prior to 2016)

### Data Context: Frequency & Severity

Insurance pricing uses **Pure Premium = Frequency × Severity**.

- **Frequency Data**: `freMTPL2freq` — one row per policy. Columns: IDpol, ClaimNb, Exposure, Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region. Fetch via `fetch_openml(data_id=41214)`.
- **Severity Data**: `freMTPL2sev` — one row per **claim** (not per policy). Columns: IDpol, ClaimAmount. Fetch via `fetch_openml(data_id=41215)`.
- **Important join logic**: Aggregate `ClaimAmount` by `IDpol` (sum) before joining to frequency table. Policies with zero claims have no severity row — these are excluded from the severity model but included in the frequency model.

### Target Directory Structure

```text
risktec_portfolio/
├── data/                  # raw data files
├── dbt_project/           # staging models, marts, and data governance tests
├── notebooks/             # EDA and model training notebooks
├── app/                   # Streamlit dashboard
├── requirements.txt
└── README.md              # Project overview with architecture diagram (Mermaid)
```

---

## 5. Execution Phases

### Phase 1: Data Ingestion & Governance Pipeline (~3 hrs)

**Objective:** Build a governed, tested data pipeline — the core "Data Engineer" deliverable.

**Tasks:**

1. Load raw data into DuckDB tables (simulating bordereaux intake into a cloud warehouse).
2. Initialize a dbt project with DuckDB adapter.
3. Create **staging models**:
   - `stg_policies`: Clean types, standardize column names from frequency data.
   - `stg_claims`: Aggregate claim amounts by policy ID from severity data.
4. Create a **mart model**:
   - `mart_pricing`: Join policies with aggregated claims. Add derived columns: `has_claim` flag, `avg_claim_amount`, `total_claim_amount`. This is the ML-ready table.
5. Implement **dbt tests** (framed as governance controls):
   - `Exposure > 0`, unique `IDpol`, `ClaimNb >= 0`, `ClaimAmount >= 0`.
   - Referential integrity between staging and mart.
6. Document all models with dbt docs (descriptions, column-level documentation).

**Output:** A repeatable `dbt build` that produces a clean, tested, documented data mart.

---

### Phase 2: Risk Pricing Models (~3 hrs)

**Objective:** Train interpretable models that predict claim frequency, severity, and pure premium.

**Tasks:**

1. **EDA Notebook**: Explore the mart data. Visualize claim frequency distribution, severity tail behavior, exposure by region, BonusMalus distribution, and feature correlations. Document key findings.
2. **Frequency Model**: Train an XGBoost model to predict `ClaimNb` using policy features with `Exposure` as a sample weight. Evaluate with Poisson deviance.
3. **Severity Model**: Train an XGBoost model on policies with claims (`ClaimNb > 0`) to predict average `ClaimAmount`. Evaluate with Gamma deviance.
4. **Pure Premium**: Combine: `predicted_frequency × predicted_severity = pure_premium`. This is the key business output.
5. **SHAP Explanations**: Generate SHAP summary and waterfall plots for both models. Save plots and trained models to `data/models/`.

**Output:** Trained models + SHAP plots + an EDA notebook showing analytical thinking.

---

### Phase 3: Streamlit Underwriting Dashboard (~3 hrs)

**Objective:** Build a visual tool that turns the AI models into business value for underwriters.

**Tasks:**

1. **Underwriting Copilot Tab**:
   - Input form: policyholder attributes (driver age, vehicle power, region, BonusMalus, etc.).
   - On submit: load trained models, predict frequency + severity + pure premium.
   - Display SHAP waterfall plot explaining _why_ this premium was recommended (e.g., "Premium increased due to high BonusMalus and urban region").
2. **Portfolio Overview Tab**:
   - Loss ratio analysis by region and driver age bucket (charts).
   - Exposure concentration by region (bar chart or simple heatmap).
   - Expected vs. actual claims comparison.
   - Highlight under-priced vs. over-priced segments.
3. **Professional polish**:
   - risktec-inspired branding/colors.
   - Clear section headers and tooltips explaining insurance terms for non-technical users.

**Output:** A running Streamlit app that an interviewer can interact with.

---

## 6. README & Presentation Assets

After all phases, generate:

1. **README.md** with:
   - Business context (1 paragraph).
   - Architecture diagram (Mermaid): `Raw Data → DuckDB → dbt Staging → dbt Mart → XGBoost Models → Streamlit`.
   - Setup instructions (`pip install`, `dbt build`, `streamlit run`).
   - Screenshots of the dashboard.
2. A **"Future Enhancements"** section listing what a production version would add (use this as interview talking points):
   - GLM baselines (Poisson frequency, Gamma severity) for actuarial benchmarking.
   - Reinsurance treaty clause extraction using RAG/LLM (Snowflake Cortex).
   - Cloud deployment to Snowflake with Dagster orchestration.
   - FastAPI model serving layer for production integration.
   - CI/CD pipeline with GitHub Actions.
