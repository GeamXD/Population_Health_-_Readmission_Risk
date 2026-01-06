# Population Health & Readmission Risk
End-to-end analytics pipeline to predict 30-day readmission risk from CMS claims data and deliver actionable population stratification for care management.

## Business problem
- Unplanned 30-day readmissions drive avoidable cost and lower quality scores.
- Care management teams need a ranked list of high-risk members with explainable drivers.
- Leadership needs measurable financial impact (expected savings, ROI) from targeted interventions.

## Data & tech stack
- **Data**: CMS claims (inpatient/outpatient/professional), eligibility/member, provider, diagnosis/procedure codes, utilization history.
- **Platform**: **Databricks** (Delta Lake, Jobs/Workflows), **Spark**, **MLflow**.
- **Transformation**: **dbt** (models + tests), Medallion design (**Bronze/Silver/Gold**).
- **Orchestration**: Databricks Workflows (or scheduled Jobs).
- **Visualization**: Databricks SQL dashboards / BI tool (Power BI / Tableau).
- **Versioning/CI**: GitHub + optional CI (lint/test/dbt).

## Architecture diagram

```mermaid
flowchart LR
  A["Raw CMS Claims<br/>(CSV, Parquet)"] --> B["Bronze Delta tables"]
  B --> C["Silver: cleaned & conformed"]
  C --> D["Gold analytics marts"]
  D --> E["Feature store / training dataset"]
  E --> F["Model training (MLflow)"]
  F --> G["Batch scoring"]
  G --> H["Risk stratification & cohorts"]
  H --> I["Dashboards / exports"]
```

## Key outputs
- **Gold tables / marts**
  - Member-level risk features and utilization history
  - Readmission label + prediction-ready training set
- **Models**
  - 30-day readmission risk classifier (AUC/PR tracked in MLflow)
  - Feature importance / explainability artifacts
- **Dashboards**
  - Population risk distribution and high-risk member lists
  - Cost impact and ROI estimates for intervention scenarios

## Link to Business Presentation (critical)
- **Business deck (Canva)**: ![Deck](https://www.canva.com/design/DAG9i6gBNGU/uWBBuT-7blUsqGmbvIYwmg/edit?utm_content=DAG9i6gBNGU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## How to run (optional)
1. Configure Databricks workspace + cluster policies.
2. Load raw claims/eligibility files into the **Bronze** landing area.
3. Run **dbt** models to build **Silver**/**Gold** tables.
4. Execute training notebook/job to register the model in **MLflow**.
5. Run batch scoring to generate risk cohorts and refresh dashboards.

> Notes: Add environment variables/secrets for storage credentials and workspace tokens as needed.
