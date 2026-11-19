# Group G074 — MP280

## Problem: City Bus Delay (Dehradun)

**Client:** Dehradun City Bus Service  
**Faculty:** Dhananjay Kumar | **Batch:** 64

---

### Team Members

| Enrolment No. | Name | Role |
|---|---|---|
| **S24CSEU2705** | Samanyu Khanna | Data Lead |
| **S24CSEU1012** | Aaditya Suri | Model Lead |
| **S24CSEU2385** | Pulkit Chahar | Service Lead |
| **S24CSEU2378** | Vishal Yadav | Ops Lead |

---

### Project Overview
An end-to-end MLOps pipeline for predicting bus arrival delays (`delay_min`) in Dehradun.
- **Data Versioning:** DVC (`data/raw/train.csv.dvc`, `data/raw/later.csv.dvc`)
- **Quality Assurance & Diagnostics:** Audit and remediation of 7 data anomalies in `src/data/inspect_issues.py`
- **Data Pipeline:** Automated cleaning pipeline in `src/data/clean.py` and `dvc.yaml`
- **Experiment Tracking:** MLflow (`mlflow.db`)
- **Prediction Service:** FastAPI (`app/main.py`) containerized with Docker & deployed via Kubernetes
- **Monitoring & Drift:** Population Stability Index (PSI) drift monitoring on `later.csv`
