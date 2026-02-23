# Bank Marketing Prediction — Coursework Report

**Module:** Machine Learning + Cloud Computing  
**Dataset:** UCI Bank Marketing Dataset (Portuguese bank telemarketing)  
**Objective:** Predict client subscription to a term deposit (`y`: yes/no)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Task A — Preprocessing](#2-task-a--preprocessing)
3. [Task B — Model Building (LR & SVM)](#3-task-b--model-building)
4. [Task C — Discussion & Findings](#4-task-c--discussion--findings)
5. [Task D — AWS Deployment](#5-task-d--deployment)
6. [Task E — Solution & Deployment Architecture](#6-task-e--architecture-diagrams)
7. [Task F — AWS Hosting Justification](#7-task-f--hosting-justification)
8. [Task G — CI/CD Pipeline](#8-task-g--cicd-pipeline)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## 1. Introduction

This project addresses a real-world binary classification problem: predicting whether a client of a Portuguese bank will subscribe to a term deposit after a telemarketing phone call. The dataset contains 41,188 records with 20 input features spanning client demographics, campaign contact details, and macroeconomic indicators.

**Key challenge:** The dataset is heavily imbalanced (~88.7% "no" vs 11.3% "yes"), and the `duration` feature introduces a data leakage risk that must be carefully managed.

---

## 2. Task A — Preprocessing

### 2.1 Missing Values

The dataset contains **no explicit null values**. However, several categorical features contain `"unknown"` entries:

| Feature    | Unknown Count | Percentage |
|-----------|--------------|------------|
| `job`      | 330          | 0.80%      |
| `marital`  | 80           | 0.19%      |
| `education`| 1,731        | 4.20%      |
| `default`  | 8,597        | 20.87%     |
| `housing`  | 990          | 2.40%      |
| `loan`     | 990          | 2.40%      |

**Decision:** Treat `"unknown"` as a valid category rather than imputing.  
**Rationale:**  
- `default` has 20.87% unknowns — too many to drop without significant data loss.  
- The bank deliberately recorded "unknown"; it carries informational value.  
- `OneHotEncoder` creates a separate binary indicator for the "unknown" category.

### 2.2 Outlier Analysis

Box plot analysis revealed outliers in `age`, `campaign`, `duration`, `pdays`, and `previous`. Using the IQR method:

**Decision:** Retain all outliers.  
**Rationale:**  
- Outliers represent genuine client behaviour (e.g., an elderly client, many contact attempts).  
- `StandardScaler` mitigates the influence of extreme values on distance-based models (SVM).  
- Removing outliers from an already imbalanced dataset would further reduce minority-class samples.

### 2.3 Feature Encoding

- **Categorical features** → `OneHotEncoder(handle_unknown='ignore')` to handle unseen categories at deployment.
- **Numeric features** → `StandardScaler()` to normalise to zero mean, unit variance.
- **Implementation:** `sklearn.compose.ColumnTransformer` inside a `Pipeline` ensures identical preprocessing during training and inference.

### 2.4 Scaling Effects

Before scaling, numeric features have vastly different ranges (e.g., `age`: 17–98 vs `nr.employed`: 4,964–5,228). After `StandardScaler`, all numeric features are centred at 0 with standard deviation 1. This is critical for SVM (distance-sensitive) and beneficial for Logistic Regression (gradient convergence).

*See notebook figures: `scaling_effect_age.png` and `scaling_effect_all.png`.*

---

## 3. Task B — Model Building

### Train/Test Split
- **Ratio:** 80% train / 20% test
- **Stratification:** Applied (`stratify=y`) to preserve the 88.7/11.3 class ratio in both sets.

### 3.1 Logistic Regression

```python
LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
```

- `class_weight='balanced'` adjusts sample weights inversely proportional to class frequencies.
- `max_iter=1000` ensures convergence.

### 3.2 Support Vector Machine

```python
SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42, C=1.0)
```

- RBF kernel captures non-linear decision boundaries.
- `probability=True` enables `predict_proba` for ROC-AUC computation.
- `class_weight='balanced'` handles imbalance.

### 3.3 Model Variants

Three feature sets were tested for each model:

| Variant | Features | Purpose |
|---------|----------|---------|
| **Benchmark** | All 20 (including `duration`) | Demonstrate duration's influence |
| **Realistic** | 19 (excluding `duration`) | Fair comparison without leakage |
| **Deployment** | 7 (form fields only) | Production-ready model |

### 3.4 Evaluation Metrics

For each model variant, we report:
- **Confusion Matrix** — True/False Positives/Negatives
- **Precision** — Of predicted "yes", how many are correct
- **Recall** — Of actual "yes", how many are detected
- **F1 Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Area under the receiver operating characteristic curve

*See notebook figures: `confusion_matrices.png` and `roc_curves.png`.*

---

## 4. Task C — Discussion & Findings

### 4.1 Duration Leakage

The `duration` feature has the **highest correlation** with the target variable. Benchmark models (with duration) achieve significantly higher performance than realistic models, confirming the dataset authors' warning.

> ⚠️ **`duration` is NOT known before the call.** After the call, `y` is already known. Including `duration` in a production model constitutes **data leakage** — using future information to predict the present.

The benchmark models exist solely to demonstrate this effect. **Only realistic/deployment models are valid for production.**

### 4.2 LR vs SVM Comparison

| Criterion | Logistic Regression | SVM (RBF Kernel) |
|-----------|--------------------|--------------------|
| Training Speed | Fast (~seconds) | Slow (~minutes on 41K rows) |
| Inference Speed | Instant | Slower (support vector computations) |
| Interpretability | High (coefficients) | Low (kernel space) |
| Non-linearity | Linear boundary only | Non-linear boundaries |
| Model Size | ~10 KB | ~50+ MB (stores support vectors) |
| Scalability | O(n) | O(n²) to O(n³) |

**Conclusion:** For this deployment, **Logistic Regression** is preferred due to comparable performance, faster inference, smaller model size, and better interpretability — all critical for a web-based POC.

### 4.3 Class Imbalance

- **Ratio:** ~7.9:1 (no:yes)
- Without handling: Models predict "no" for all inputs → 88.7% accuracy but 0% recall.
- **Mitigation:** `class_weight='balanced'` in both models, which penalises misclassification of the minority class proportionally.
- **Metric choice:** F1 and Recall are prioritised over accuracy.

### 4.4 Limitations

1. Deployment model uses only 7 client-profile features, limiting predictive power.
2. No hyperparameter tuning performed (out of scope; default parameters used).
3. SVM training is computationally expensive for larger datasets.
4. In a real-world system, campaign-context and economic features would be injected server-side.

---

## 5. Task D — Deployment

### Web Application Design

The POC is a **Flask web application** that:
1. Presents a **form** with exactly **7 input fields**: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`.
2. Submits the form to a `/predict` endpoint.
3. Runs the input through the saved `sklearn Pipeline` (preprocessing + Logistic Regression).
4. Displays the prediction: **"Client subscribed a term deposit: YES / NO"** with confidence score.

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML/CSS (Jinja2 templates) |
| Backend | Flask + Gunicorn |
| ML Pipeline | scikit-learn Pipeline (StandardScaler + OneHotEncoder + LogisticRegression) |
| Container | Docker (multi-stage build) |
| Cloud | AWS EC2 (t2.micro / t3.micro) |
| Registry | Amazon ECR |
| CI/CD | GitHub Actions |

---

## 6. Task E — Architecture Diagrams

### 6.1 Solution Architecture Diagram (Request + Data Flows)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SOLUTION ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐     REQUEST FLOW                                          │
│  │  Client   │─────────────────────────────────────────────┐            │
│  │ (Browser) │                                             │            │
│  └────┬─────┘                                              │            │
│       │ 1. HTTP GET /                                      │            │
│       ▼                                                    │            │
│  ┌───────────────┐    2. Returns HTML Form                 │            │
│  │   Flask App    │◄───────────────────────────────────────┘            │
│  │   (app.py)     │                                                     │
│  └───────┬───────┘                                                     │
│          │                                                              │
│          │ 3. HTTP POST /predict                                        │
│          │    {age, job, marital, education,                             │
│          │     default, housing, loan}                                   │
│          ▼                                                              │
│  ┌───────────────────────────────────────────────────────┐              │
│  │              PREDICTION PIPELINE                       │              │
│  │  ┌─────────────────┐    ┌──────────────────────────┐  │              │
│  │  │  ColumnTransform │    │  Logistic Regression     │  │              │
│  │  │  ┌─────────────┐│    │  (trained classifier)    │  │              │
│  │  │  │StandardScale││    │                          │  │              │
│  │  │  │ (age)       ││───▶│  predict(X) → 0 or 1    │  │              │
│  │  │  ├─────────────┤│    │  predict_proba(X) →      │  │              │
│  │  │  │OneHotEncoder ││    │    [P(no), P(yes)]      │  │              │
│  │  │  │ (6 cat cols)││    └──────────────────────────┘  │              │
│  │  │  └─────────────┘│                                  │              │
│  │  └─────────────────┘                                  │              │
│  └───────────────────────────────────────────────────────┘              │
│          │                                                              │
│          │ 4. Response: YES/NO + confidence                             │
│          ▼                                                              │
│  ┌──────────┐                                                          │
│  │  Client   │ ◄── result.html rendered with prediction                 │
│  │ (Browser) │                                                          │
│  └──────────┘                                                          │
│                                                                         │
│  DATA FLOW ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─               │
│                                                                         │
│  ┌────────────┐  train   ┌─────────────┐  export   ┌────────────────┐  │
│  │  UCI Bank   │────────▶│  Notebook    │─────────▶│  .joblib Model  │  │
│  │  Marketing  │         │  (sklearn    │          │  Pipeline       │  │
│  │  Dataset    │         │   Pipeline)  │          │  (preprocessor  │  │
│  │  (41,188    │         │             │          │   + classifier) │  │
│  │   records)  │         └─────────────┘          └───────┬────────┘  │
│  └────────────┘                                           │            │
│                                                           │ loaded at  │
│                                                           │ startup    │
│                                                           ▼            │
│                                                    ┌──────────────┐    │
│                                                    │  Flask App   │    │
│                                                    │  (app.py)    │    │
│                                                    └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Deployment Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AWS DEPLOYMENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐                                                          │
│  │ Internet  │                                                          │
│  │  Users    │                                                          │
│  └─────┬────┘                                                          │
│        │ HTTPS (port 443)                                               │
│        ▼                                                                │
│  ┌───────────────────────────────────────────┐                          │
│  │        AWS CLOUD (eu-west-1)              │                          │
│  │                                           │                          │
│  │  ┌───────────────────────────────────┐    │                          │
│  │  │  Security Group                    │    │                          │
│  │  │  • Inbound: 80, 443 (HTTP/HTTPS)  │    │                          │
│  │  │  • Inbound: 22 (SSH — admin only)  │    │                          │
│  │  │  • Outbound: All                   │    │                          │
│  │  │                                    │    │                          │
│  │  │  ┌──────────────────────────────┐  │    │                          │
│  │  │  │  EC2 Instance (t3.micro)     │  │    │                          │
│  │  │  │  Amazon Linux 2023           │  │    │                          │
│  │  │  │                              │  │    │                          │
│  │  │  │  ┌────────────────────────┐  │  │    │                          │
│  │  │  │  │  Docker Container      │  │  │    │                          │
│  │  │  │  │                        │  │  │    │                          │
│  │  │  │  │  ┌──────────────────┐  │  │  │    │                          │
│  │  │  │  │  │  Gunicorn (WSGI) │  │  │  │    │                          │
│  │  │  │  │  │  Workers: 2      │  │  │  │    │                          │
│  │  │  │  │  │       │          │  │  │  │    │                          │
│  │  │  │  │  │  ┌────▼───────┐  │  │  │  │    │                          │
│  │  │  │  │  │  │ Flask App  │  │  │  │  │    │                          │
│  │  │  │  │  │  │ (app.py)   │  │  │  │  │    │                          │
│  │  │  │  │  │  │     │      │  │  │  │  │    │                          │
│  │  │  │  │  │  │ ┌───▼────┐ │  │  │  │  │    │                          │
│  │  │  │  │  │  │ │ Model  │ │  │  │  │  │    │                          │
│  │  │  │  │  │  │ │.joblib │ │  │  │  │  │    │                          │
│  │  │  │  │  │  │ └────────┘ │  │  │  │  │    │                          │
│  │  │  │  │  │  └────────────┘  │  │  │  │    │                          │
│  │  │  │  │  └──────────────────┘  │  │  │    │                          │
│  │  │  │  │   Port: 5000 → 80     │  │  │    │                          │
│  │  │  │  └────────────────────────┘  │  │    │                          │
│  │  │  └──────────────────────────────┘  │    │                          │
│  │  └───────────────────────────────────┘    │                          │
│  │                                           │                          │
│  │  ┌───────────────────────────────────┐    │                          │
│  │  │  Amazon ECR                        │    │                          │
│  │  │  (Container Registry)              │    │                          │
│  │  │  bank-marketing-prediction:latest  │    │                          │
│  │  └───────────────────────────────────┘    │                          │
│  │                                           │                          │
│  └───────────────────────────────────────────┘                          │
│                                                                         │
│  ┌───────────────────────────────────────────┐                          │
│  │  GitHub Repository                        │                          │
│  │  ├── .github/workflows/ci-cd.yml          │                          │
│  │  ├── Dockerfile                           │                          │
│  │  ├── app.py + model/ + templates/         │                          │
│  │  └── GitHub Actions (CI/CD)               │                          │
│  └───────────────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Task F — AWS Hosting Justification

### Chosen Approach: **Amazon EC2**

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **EC2** | Full control, simple Docker hosting, persistent, easy SSH debugging | Manual scaling, OS maintenance | ✅ **Selected** |
| Lambda | Serverless, pay-per-use, auto-scaling | Cold starts (~5s for ML model), 250MB package limit, 15min timeout | ❌ Not ideal |
| ECS | Managed containers, auto-scaling, load balancing | Over-engineered for a POC, higher complexity | ❌ Over-engineered |

### Why EC2 is Appropriate for This POC

1. **Simplicity:** A single `t3.micro` instance runs the entire stack (Docker + Flask + model). No orchestration overhead.
2. **Free Tier:** `t3.micro` is eligible for the AWS Free Tier (750 hours/month for 12 months).
3. **Persistent:** Unlike Lambda, the model is loaded once at startup and stays in memory — no cold start penalty.
4. **Full Control:** SSH access for debugging, custom Docker setup, and straightforward security group configuration.
5. **Sufficient for POC:** A proof-of-concept does not require auto-scaling or load balancing. A single instance handles the expected low traffic.
6. **Docker Support:** EC2 natively supports Docker, matching our containerised architecture.

### EC2 Deployment Steps

```bash
# 1. Launch EC2 instance (Amazon Linux 2023, t3.micro)
#    - Security Group: Allow inbound 80, 443, 22

# 2. SSH into the instance
ssh -i key.pem ec2-user@<EC2_PUBLIC_IP>

# 3. Install Docker
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# 4. Install AWS CLI (for ECR access)
sudo yum install -y aws-cli

# 5. Login to ECR
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.eu-west-1.amazonaws.com

# 6. Pull and run the container
docker pull <ACCOUNT_ID>.dkr.ecr.eu-west-1.amazonaws.com/bank-marketing-prediction:latest

docker run -d \
  --name bank-marketing-app \
  --restart unless-stopped \
  -p 80:5000 \
  <ACCOUNT_ID>.dkr.ecr.eu-west-1.amazonaws.com/bank-marketing-prediction:latest

# 7. Verify
curl http://localhost/health
```

---

## 8. Task G — CI/CD Pipeline

### 8.1 CI/CD Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CI/CD PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    git push     ┌──────────────────────────┐              │
│  │ Developer │───────────────▶│  GitHub Repository       │              │
│  └──────────┘  (main branch)  │  (Source Code + Model)   │              │
│                               └──────────┬───────────────┘              │
│                                          │                              │
│                                          │ Trigger: push to main        │
│                                          ▼                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    GITHUB ACTIONS WORKFLOW                         │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  JOB 1: LINT & TEST                                         │  │  │
│  │  │                                                             │  │  │
│  │  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │  │  │
│  │  │  │ Checkout │──▶│ Setup    │──▶│ Install  │──▶│ Flake8  │ │  │  │
│  │  │  │  Code    │   │ Python   │   │  Deps    │   │  Lint   │ │  │  │
│  │  │  │          │   │ 3.11     │   │          │   │         │ │  │  │
│  │  │  └──────────┘   └──────────┘   └──────────┘   └────┬────┘ │  │  │
│  │  │                                                     │      │  │  │
│  │  │                    ┌──────────┐   ┌──────────────┐  │      │  │  │
│  │  │                    │  Pytest  │◀──│ Model Load   │◀─┘      │  │  │
│  │  │                    │  Tests   │   │ Verification │         │  │  │
│  │  │                    └────┬─────┘   └──────────────┘         │  │  │
│  │  └─────────────────────────┼───────────────────────────────────┘  │  │
│  │                            │ ✅ Pass                              │  │
│  │                            ▼                                      │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  JOB 2: BUILD & PUSH                                       │  │  │
│  │  │                                                             │  │  │
│  │  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │  │  │
│  │  │  │ AWS      │──▶│ ECR      │──▶│ Docker   │──▶│ Push to │ │  │  │
│  │  │  │ Creds    │   │ Login    │   │ Build    │   │  ECR    │ │  │  │
│  │  │  │ Config   │   │          │   │ (multi-  │   │ :latest │ │  │  │
│  │  │  │          │   │          │   │  stage)  │   │ :sha    │ │  │  │
│  │  │  └──────────┘   └──────────┘   └──────────┘   └────┬────┘ │  │  │
│  │  └─────────────────────────────────────────────────────┼──────┘  │  │
│  │                                                        │ ✅      │  │
│  │                                                        ▼         │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  JOB 3: DEPLOY                                             │  │  │
│  │  │                                                             │  │  │
│  │  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │  │  │
│  │  │  │ SSH to   │──▶│ Pull     │──▶│ Stop Old │──▶│ Start   │ │  │  │
│  │  │  │ EC2      │   │ Latest   │   │ Container│   │ New     │ │  │  │
│  │  │  │          │   │ Image    │   │          │   │Container│ │  │  │
│  │  │  └──────────┘   └──────────┘   └──────────┘   └────┬────┘ │  │  │
│  │  │                                                     │      │  │  │
│  │  │                              ┌──────────────────┐   │      │  │  │
│  │  │                              │  Health Check    │◀──┘      │  │  │
│  │  │                              │  GET /health     │          │  │  │
│  │  │                              │  ✅ 200 OK       │          │  │  │
│  │  │                              └──────────────────┘          │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### 8.2 CI/CD Process Description

The CI/CD pipeline is implemented using **GitHub Actions** and triggered on every push to the `main` branch.

#### Stage 1: Lint & Test (Continuous Integration)

| Step | Action | Details |
|------|--------|---------|
| 1 | **Checkout** | Clone the repository |
| 2 | **Setup Python** | Install Python 3.11 |
| 3 | **Install Dependencies** | `pip install -r requirements.txt` + linters/test tools |
| 4 | **Lint** | `flake8 app.py` — enforce code style |
| 5 | **Unit Tests** | `pytest tests/` — run any available tests |
| 6 | **Model Verification** | Attempt to load `.joblib` model to verify integrity |

**Gate:** If any step fails, the pipeline stops. No broken code is deployed.

#### Stage 2: Build & Push (Continuous Delivery)

| Step | Action | Details |
|------|--------|---------|
| 1 | **Configure AWS** | Set IAM credentials from GitHub Secrets |
| 2 | **ECR Login** | Authenticate Docker to Amazon ECR |
| 3 | **Docker Build** | Multi-stage build from `Dockerfile` |
| 4 | **Tag & Push** | Push image tagged with commit SHA + `latest` to ECR |

#### Stage 3: Deploy (Continuous Deployment)

| Step | Action | Details |
|------|--------|---------|
| 1 | **SSH to EC2** | Connect via `appleboy/ssh-action` using stored SSH key |
| 2 | **Pull Image** | `docker pull` the latest image from ECR |
| 3 | **Stop Old Container** | `docker stop` + `docker rm` the running container |
| 4 | **Start New Container** | `docker run -d -p 80:5000 ...` |
| 5 | **Health Check** | `curl /health` to verify the new container is serving |

#### GitHub Secrets Required

| Secret | Purpose |
|--------|---------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `EC2_HOST` | Public IP/DNS of the EC2 instance |
| `EC2_SSH_KEY` | Private SSH key for EC2 access |

---

## 9. Conclusion

This project demonstrates a complete ML pipeline from data preprocessing through model deployment:

1. **Preprocessing:** Handled implicit missing values ("unknown"), retained meaningful outliers, applied StandardScaler + OneHotEncoder via a reusable Pipeline.
2. **Modelling:** Built and compared Logistic Regression and SVM, correctly addressing the `duration` leakage issue with benchmark vs realistic models.
3. **Deployment:** Selected Logistic Regression for deployment due to superior efficiency, deployed as a containerised Flask application on AWS EC2.
4. **CI/CD:** Automated the entire build-test-deploy cycle with GitHub Actions.

The POC form accepts 7 client-profile features and returns a subscription prediction with confidence, fulfilling all coursework requirements.

---

## 10. References

1. Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22-31.
2. UCI Machine Learning Repository — Bank Marketing Dataset. https://archive.ics.uci.edu/dataset/222/bank+marketing
3. scikit-learn Documentation — Pipeline, ColumnTransformer. https://scikit-learn.org/stable/
4. AWS EC2 Documentation. https://docs.aws.amazon.com/ec2/
5. Docker Documentation. https://docs.docker.com/
6. GitHub Actions Documentation. https://docs.github.com/en/actions
