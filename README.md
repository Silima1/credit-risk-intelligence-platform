```md
# ABI Credit Risk Intelligence Platform (FastAPI + Vite)

End-to-end platform (backend + frontend) for **credit risk analysis** featuring:
- **Prediction by Application ID** (risk probability + decision)
- **XAI report per application** (SHAP explanations with human-readable narrative)
- **Executive Dashboard** (aggregated metrics + drift signals)
- **Threshold Optimization** (cost-sensitive decisioning)
- **Modern Frontend** (Vite + React consuming the API)

---

## Architecture

```

abi_dashboard/
backend/
app/
main.py
core/
services/
schemas/
requirements.txt
Dockerfile
frontend/
src/
api/
components/
package.json
Dockerfile
docker-compose.yml
README.md

```

---

## Requirements

### Without Docker
- **Python 3.10**
- **Node.js 18+**
- (Optional) `venv`

### With Docker
- Docker Engine
- Docker Compose v2 (`docker compose`)

---

## Data & Model

The backend loads:
- `train.csv`
- `test.csv`
- `rf_kaggle.joblib` (trained Random Forest pipeline)

**Recommended structure**
```

backend/source/
train.csv
test.csv
rf_kaggle.joblib

````

Make sure `model_path` (settings/config) points to this file.

---

## Configuration Notes (Backend)

If you use `settings.model_path`, ensure it matches:

- **Local run**:  
  `backend/source/rf_kaggle.joblib`

- **Docker** (inside container):  
  `/app/source/rf_kaggle.joblib`  
  (when copied into the image)

---

# Run WITHOUT Docker (Local)

## 1) Backend

```bash
cd backend
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m uvicorn app.main:app --reload --port 8000
````

Backend URLs:

* API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Docs (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend:

* [http://localhost:5173](http://localhost:5173)

> The frontend assumes the backend at `http://127.0.0.1:8000`.
> Adjust `frontend/src/api/client.ts` if needed.

---

# Run WITH Docker

## 1) Build & Start

From the project root:

```bash
docker compose up --build
```

Services:

* Frontend → [http://localhost:5173](http://localhost:5173)
* Backend  → [http://localhost:8000](http://localhost:8000)

---

## 2) Stop

```bash
docker compose down
```

---

## 3) Clean rebuild (recommended after dependency changes)

```bash
docker compose down -v
docker compose build --no-cache backend
docker compose up
```

---

# Main Backend Endpoints

## Health & Meta

* `GET /health` → service status
* `GET /meta` → available features and `application_id` presence

## Applications

* `GET /applications?limit=100` → list of Application IDs

## Prediction

* `POST /predict`
  (by `application_id` or raw `features`)

  * returns: `risk_probability`, `prediction`, `recommendation`, `used_threshold`

## XAI (Explainability)

* `POST /xai/local` → local explanation for a single application
* `GET /xai/global?sample_size=200` → global feature importance

## Dashboard

* `GET /dashboard?sample_size=500` → aggregated metrics + score drift

## Drift

* `GET /drift/score?sample_size=200` → PSI / KS drift metrics

## Optimization

* `POST /optimize/threshold` → optimal decision threshold given cost matrix

---

# Troubleshooting

## Frontend: `vite: not found`

```bash
cd frontend
npm install
npm run dev
```

---

## Frontend: `Failed to resolve import "recharts"`

Install missing dependency:

```bash
cd frontend
npm install recharts
```

---

## Docker: backend crashes on startup (`imblearn` / `_safe_tags`)

This is a **version mismatch** between `imbalanced-learn` and `scikit-learn`.

**Recommended fix** (pin compatible versions in `backend/requirements.txt`):

```txt
scikit-learn==1.4.2
imbalanced-learn==0.12.3
```

Then rebuild:

```bash
docker compose down -v
docker compose build --no-cache backend
docker compose up
```

---

## Docker permission denied (Linux)

```bash
sudo usermod -aG docker $USER
newgrp docker
docker ps
```

---

# Production Note

Current `docker-compose.yml` is **development-oriented**:

* Frontend runs with `vite --host`
* Backend uses `uvicorn`

For production, recommended:

* Build frontend (`npm run build`)
* Serve via **Nginx**
* Run backend with `uvicorn` (no reload) behind a reverse proxy

 
AUTHOR: Leonel Silima