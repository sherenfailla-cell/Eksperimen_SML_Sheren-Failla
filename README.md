# Telco Customer Churn — Experiment & Automated Preprocessing

Part of an end-to-end MLOps pipeline (experiment → tracked training → CI retraining → monitored serving).

**This repo covers the experimentation stage:**
- `preprocessing/Eksperimen_Sheren-Failla.ipynb` — EDA & preprocessing experiment on 7,043 telco customers (IBM sample dataset). Key findings: hidden missing values in `TotalCharges` (new customers with zero tenure), strong churn signal from month-to-month contracts (~43% churn vs <12% on long contracts), imbalanced target (26.5%).
- `preprocessing/automate_Sheren-Failla.py` — the notebook converted into a reusable pipeline (clean → encode → stratified split → leakage-free scaling), runnable headless.
- `.github/workflows/preprocessing.yml` — GitHub Actions workflow that re-runs preprocessing on every push.

**Related repo:** CI retraining pipeline → `Workflow-CI-Telco-Churn`
