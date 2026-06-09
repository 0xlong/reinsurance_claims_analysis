# risktec — Data & AI Engineer — real-talk analysis

**Location:** Zurich, hybrid, 80–100% (Swiss workweek — 100% = full-time, 80% = 4 days/week)
**Company:** risktec, founded 2022, ~50 people, boutique consulting firm for re/insurance + critical infra + public sector

---

## Who they're actually looking for

A **senior data engineer who can also do ML / AI work end-to-end**, not a research-only data scientist and not a pure pipeline plumber. Think "one-person data team on a client project" — someone who can walk in, talk to a reinsurance underwriter, design the data model, build the pipeline in Snowflake, ship an ML model, and then operate it.

Because it's a boutique + consulting shop, you will be:
- Client-facing (reinsurers, insurers, infrastructure providers, public sector)
- Wearing many hats (engineer + analyst + sometimes architect)
- Expected to deliver things, not write strategy decks
- Probably working on multiple client projects in parallel or in sequence

It's a 4-year-old growing firm, so expect startup energy — processes still being built, things move fast, scope is broad.

---

## Skills — what you really need

### Must-have (non-negotiable despite soft wording)
- **5+ years** in data engineering / data science / analytics (real production work)
- **SQL** — expert level. Non-trivial queries, window functions, optimization.
- **Python** — primary language. Pandas, scikit-learn, pipelines. (R is mentioned but Python is the reality.)
- **Snowflake** — listed first in the cloud bullet, which is a tell. Their stack almost certainly runs on it.
- **Cloud platform** — AWS or Azure. One is enough; both is better.
- **Data modeling** — dimensional modeling, star schemas, handling slowly changing dimensions. Insurance data is messy and relational.
- **Practical ML** — build, deploy, monitor models. Not paper-writing, not Kaggle.

### Strongly implied but not written
- **dbt** (standard pairing with Snowflake)
- **Orchestration** — Airflow, Dagster, or similar
- **Git + CI/CD** — they talk about "cloud-native" and "scalable architectures"
- **GenAI / LLMs** — "AI" is in the job title and they mention "AI-driven insights"; expect RAG / LLM work on contracts and claims
- **English fluency** — explicit
- **Consulting chops** — explaining technical work to non-technical insurance people

### Genuinely nice-to-have
- **German** — Zurich, Swiss/German clients. Written as optional but will help day-to-day.
- **Re/insurance domain knowledge** — actuarial basics, pricing, reserving, catastrophe modeling, reinsurance treaties. If you don't have it, they'll teach you, but it's a steep ramp.

---

## What you'll actually do day-to-day

Decoding the corporate bullets:

| They say | It means |
|---|---|
| "Design advanced data models" | Build star/snowflake schemas for policy, claims, exposure, treaty data in Snowflake |
| "AI-driven insights for underwriting, risk, contracts, claims, portfolio steering" | Pricing models, claims triage/fraud detection, contract-clause extraction (likely LLM-based), portfolio optimization dashboards |
| "Integrate and process large-scale re/insurance datasets" | ETL/ELT — ingest messy CSVs, bordereaux, broker files, legacy system exports; clean them; land them in Snowflake |
| "Ensure data quality and governance" | Write tests, lineage, documentation — the unglamorous 40% of the job |
| "Work with cloud platforms (Snowflake, AWS, Azure)" | Live in Snowflake; touch AWS/Azure for storage, compute, orchestration |
| "Collaborate with teams to deliver data-driven products" | Client delivery work — workshops, requirements, demos, handover |

Concrete project examples you'll likely touch:
- Pricing engine for a specialty line of business
- Catastrophe/exposure data aggregation
- Claims fraud or severity models
- Contract clause extraction with LLMs (reinsurance treaties are text-heavy)
- Portfolio dashboards for underwriting leadership
- Migration from legacy on-prem DW to Snowflake

---

## Red / yellow flags to notice

- **"Boutique + consulting"** → client-facing, possibly travel (Zurich HQ, clients across Europe), billable hours mindset
- **Very broad responsibility scope** → can be exciting or a firehose depending on staffing
- **Founded 2022, 50+ people** → fast growth, processes still forming, senior hires expected to build tooling themselves
- **"End-to-end… architecting, implementing, integrating, and operating"** → you own it in prod, not just build and hand off
- **Zurich living costs** → salary should match; negotiate accordingly
- **80–100%** workload option → positive signal (Swiss work-life flexibility is real)

---

## Who is the ideal candidate (one sentence)

A **senior full-stack data/ML engineer with 5+ years building production Snowflake + Python systems**, comfortable in front of insurance clients, who can ship a pricing or claims model end-to-end without needing a separate data engineer, MLOps person, or business analyst to hold their hand.

---

## Candidate gap analysis + GitHub projects to build

### What the candidate already has (strong)
- 5+ yrs at Tier-1 financial institutions (UBS, Roche, GE)
- Python, SQL, Snowflake (Snowflake GenAI cert is a direct match)
- dbt + Snowflake + Streamlit projects on GitHub
- Airflow cert, Looker, BigQuery/GCP
- Risk analytics at UBS (AML, payment anomaly, liquidity)
- Pricing model experience at GE (R)
- Imperial MSc Business Analytics

### Real gaps vs the risktec role

1. **Re/insurance domain** — the #1 gap. CV shows banking + pharma + energy, zero insurance. risktec mentions re/insurance ~15 times.
2. **AWS or Azure** — CV shows GCP + Snowflake. Job explicitly lists AWS/Azure.
3. **Production ML (not just analytics)** — CV is heavier on dashboards/EDA than on shipped models. Job wants pricing, fraud, claims ML.
4. **Applied LLM/RAG on documents** — has the Snowflake Cortex cert but no visible LLM project on contract/treaty text.
5. **"Portfolio steering"** — no portfolio-level analytics project visible.

### Projects to build (ranked by ROI)

**Project 1 — Insurance pricing model (biggest ROI, fills domain + ML gap)**
- Dataset: French Motor TPL (freMTPL2freq/sev), Swedish motor, or Kaggle auto insurance
- Build: frequency × severity GLMs + XGBoost Tweedie, compare; calibrated pricing output
- Stack: Python, scikit-learn/xgboost, Snowflake (data layer), dbt (feature marts), Streamlit (pricing UI)
- Why it works: directly mirrors "AI-driven insights for underwriting, risk, pricing"

**Project 2 — Reinsurance treaty clause extraction (RAG)**
- Dataset: public reinsurance treaty samples (SCOR / Munich Re investor docs, SEC filings of reinsurers), or synthetic treaty text
- Build: RAG that extracts retentions, limits, exclusions, aggregate caps, reinstatement clauses into structured JSON
- Stack: Snowflake Cortex (leverage existing cert) or LangChain + pgvector, Streamlit UI
- Why it works: reinsurance is contract-heavy; this is exactly the kind of GenAI use case a boutique pitches to clients

**Project 3 — Catastrophe exposure + portfolio analytics**
- Dataset: OpenFEMA claims, USGS earthquake, NOAA hurricane tracks + synthetic property portfolio
- Build: exposure aggregation data model (zonal), PML/AAL calculations, portfolio dashboard
- Stack: Snowflake + dbt, Python (h3/shapely for geospatial), Streamlit or Looker
- Why it works: hits "portfolio steering" and "risk analytics" head-on; differentiates from generic data engineering portfolios

**Project 4 — AWS-deployed claims fraud detection (fills cloud gap)**
- Dataset: Kaggle auto insurance fraud, or Medicare fraud
- Build: feature pipeline then model then deployed inference API + monitoring
- Stack: AWS (S3 + Glue + SageMaker OR Lambda + ECS), MLflow, Airflow for orchestration
- Why it works: closes the AWS gap AND shows end-to-end MLOps (the job says "operating solutions")

### Minimum viable plan if short on time
Ship Projects 1 and 2. They cover insurance domain + pricing + LLM/RAG, which is the core of what risktec actually sells. Project 3 is a strong differentiator. Project 4 only if AWS comes up in the interview.

### CV tweaks worth doing alongside the projects
- Title: CV header says "Data Analyst & AI Engineer" — change to "Data & AI Engineer" to mirror the JD exactly
- Reframe GE pricing model bullet to emphasize pricing model for risk analytics (insurance-adjacent language)
- Reframe UBS bullets to foreground risk analytics and data quality/governance over blockchain (risktec cares about risk, not NFTs)
- Fix typo: "Understanding fincnail markets" → "Understanding Financial Markets"

---

## Who should NOT apply

- Pure research data scientists who don't write production code
- Pipeline-only engineers with no ML exposure
- Juniors (the 5+ years is real in a boutique — no mentorship infrastructure)
- People who dislike client interaction or domain immersion
- People who want to work only on one deep technical problem — this role is broad
