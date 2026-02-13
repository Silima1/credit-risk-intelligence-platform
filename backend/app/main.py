# backend/app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from app.core.model_store import store
from app.services.predict import _to_dataframe, predict
from app.services.xai import XAIEngine
from app.services.optimize import optimal_threshold
from app.services.drift import score_drift

from app.schemas.predict import PredictRequest, PredictResponse
from app.schemas.xai import LocalExplanationResponse, GlobalImportanceResponse
from app.schemas.optimize import OptimizeThresholdRequest, OptimizeThresholdResponse
from app.schemas.drift import ScoreDriftResponse
from app.schemas.dashboard import DashboardResponse
from app.schemas.report import ReportResponse

app = FastAPI(title="Credit Risk Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

xai_engine: XAIEngine | None = None


@app.on_event("startup")
def startup():
    global xai_engine
    store.load()
    xai_engine = XAIEngine(store.model)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    return {
        "feature_cols": store.feature_cols,
        "has_application_id": "application_id" in store.test_df.columns,
    }


@app.get("/applications")
def list_applications(limit: int = 100):
    if "application_id" not in store.test_df.columns:
        return {"items": []}
    items = store.test_df["application_id"].dropna().unique().tolist()[:limit]
    return {"items": items}


@app.get("/dashboard", response_model=DashboardResponse)
def dashboard(sample_size: int = 500):
    X = store.test_df[store.feature_cols]
    n = min(sample_size, len(X))
    if n <= 0:
        raise HTTPException(status_code=400, detail="No test data available")

    Xs = X.head(n)
    proba = store.model.predict_proba(Xs)[:, 1]

    high = int((proba >= 0.7).sum())
    med = int(((proba >= 0.3) & (proba < 0.7)).sum())
    low = int((proba < 0.3).sum())

    avg = float(proba.mean())
    std = float(proba.std())

    drift_detected = False
    severity = "low"
    vpsi = 0.0
    vks = 0.0
    if len(X) >= 200:
        a = X.iloc[:100]
        b = X.iloc[100:200]
        ra = store.model.predict_proba(a)[:, 1]
        rb = store.model.predict_proba(b)[:, 1]
        out = score_drift(ra, rb)
        drift_detected = bool(out["drift_detected"])
        severity = str(out["severity"])
        vpsi = float(out["psi"])
        vks = float(out["ks"])

    return {
        "total_applications": int(n),
        "high_risk": high,
        "medium_risk": med,
        "low_risk": low,
        "avg_risk_score": avg,
        "std_risk_score": std,
        "score_drift_detected": drift_detected,
        "score_drift_severity": severity,
        "score_drift_psi": vpsi,
        "score_drift_ks": vks,
    }


@app.post("/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    if req.features is None and req.application_id is None:
        raise HTTPException(status_code=400, detail="Provide either application_id or features")

    if req.application_id is not None:
        if "application_id" not in store.test_df.columns:
            raise HTTPException(status_code=400, detail="test.csv has no application_id column")
        row = store.test_df[store.test_df["application_id"] == req.application_id]
        if len(row) == 0:
            raise HTTPException(status_code=404, detail="Application ID not found")
        X = row[store.feature_cols].head(1)
    else:
        X = _to_dataframe(store.feature_cols, req.features)

    risk, pred = predict(store.model, X)

    # Mantém comportamento antigo (APPROVE/REVIEW)
    rec = "REVIEW" if risk > req.threshold else "APPROVE"

    return PredictResponse(
        risk_probability=float(risk),
        prediction=int(pred),
        recommendation=rec,
        used_threshold=float(req.threshold),
    )


@app.post("/xai/local", response_model=LocalExplanationResponse)
def xai_local(req: PredictRequest):
    if xai_engine is None:
        raise HTTPException(status_code=500, detail="XAI engine not initialized")

    if req.application_id is not None:
        if "application_id" not in store.test_df.columns:
            raise HTTPException(status_code=400, detail="test.csv has no application_id column")
        row = store.test_df[store.test_df["application_id"] == req.application_id]
        if len(row) == 0:
            raise HTTPException(status_code=404, detail="Application ID not found")
        X = row[store.feature_cols].head(1)
    elif req.features is not None:
        X = _to_dataframe(store.feature_cols, req.features)
    else:
        raise HTTPException(status_code=400, detail="Provide application_id or features")

    items, reason_codes = xai_engine.local(X, instance_idx=0, top_n=8)
    return {"items": items, "reason_codes": reason_codes}


@app.get("/xai/global", response_model=GlobalImportanceResponse)
def xai_global(sample_size: int = 200):
    if xai_engine is None:
        raise HTTPException(status_code=500, detail="XAI engine not initialized")

    X = store.test_df[store.feature_cols].head(max(50, min(sample_size, len(store.test_df))))
    items = xai_engine.global_importance(X, top_n=20)
    return {"items": items}


@app.post("/optimize/threshold", response_model=OptimizeThresholdResponse)
def optimize_threshold(req: OptimizeThresholdRequest):
    if "fraud_flag" not in store.train_df.columns:
        raise HTTPException(status_code=400, detail="train.csv has no fraud_flag")

    X = store.train_df[store.feature_cols].head(min(5000, len(store.train_df)))
    y = store.train_df["fraud_flag"].head(len(X)).values
    y_proba = store.model.predict_proba(X)[:, 1]

    t, c = optimal_threshold(y, y_proba, req.cost_matrix.model_dump())
    return {"optimal_threshold": float(t), "expected_cost": float(c)}


@app.get("/drift/score", response_model=ScoreDriftResponse)
def drift_score(sample_size: int = 200):
    X = store.test_df[store.feature_cols]
    n = min(sample_size * 2, len(X))
    if n < 50:
        raise HTTPException(status_code=400, detail="Not enough data")

    a = X.iloc[: n // 2]
    b = X.iloc[n // 2: n]

    ra = store.model.predict_proba(a)[:, 1]
    rb = store.model.predict_proba(b)[:, 1]

    out = score_drift(ra, rb)
    return out


# ============================================================
# REPORT XAI ESPECÍFICO POR CLIENTE + GRÁFICOS
# ============================================================

@app.get("/report/{application_id}", response_model=ReportResponse)
def report(application_id: str, threshold: float = 0.5):
    if xai_engine is None:
        raise HTTPException(status_code=500, detail="XAI engine not initialized")

    if "application_id" not in store.test_df.columns:
        raise HTTPException(status_code=400, detail="test.csv has no application_id column")

    row = store.test_df[store.test_df["application_id"] == application_id]
    if len(row) == 0:
        raise HTTPException(status_code=404, detail="application_id not found")

    X = row[store.feature_cols].head(1)
    proba = float(store.model.predict_proba(X)[:, 1][0])

    # decisão (3 níveis)
    if proba >= max(0.7, threshold):
        decision = "REJECT"
    elif proba >= threshold:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    # benchmark vs população
    sample = store.test_df[store.feature_cols].head(min(3000, len(store.test_df)))
    pop_scores = store.model.predict_proba(sample)[:, 1]
    pop_avg = float(np.mean(pop_scores))
    pop_std = float(np.std(pop_scores))
    percentile = float((pop_scores < proba).mean() * 100.0)

    # narrativa específica
    diff = proba - pop_avg
    if abs(diff) < 0.02:
        comp = "muito próximo da média"
    elif diff > 0:
        comp = "acima da média (mais arriscado)"
    else:
        comp = "abaixo da média (menos arriscado)"

    narrative = (
        f"Para este pedido (ID {application_id}), o modelo estima risco {proba:.3f}. "
        f"Isto está {comp} do conjunto de dados (média {pop_avg:.3f}). "
        f"Comparado com a população, este cliente está no percentil {percentile:.1f}% "
        f"(0% = menos risco, 100% = mais risco). "
        f"Com o threshold {threshold:.2f}, a decisão recomendada é {decision}."
    )

    # XAI local + waterfall
    items, reason_codes = xai_engine.local(X, instance_idx=0, top_n=10, threshold=0.01)
    wf = xai_engine.waterfall(X, instance_idx=0, top_n=12)

    drivers = []
    for it in items[:8]:
        v = float(it["shap_value"])
        strength = "large" if abs(v) >= 0.05 else "medium" if abs(v) >= 0.02 else "small"
        drivers.append(
            {
                "title": it.get("display_feature") or it["feature"],
                "direction": "increases" if v > 0 else "decreases",
                "strength": strength,
                "shap_value": v,
                "narrative": it.get("impact_text") or "",
            }
        )

    improve = []
    for d in drivers:
        if d["direction"] == "increases":
            improve.append(f"Mitigar o impacto de: {d['title']} (aumenta o risco).")
    improve = improve[:5] if improve else ["Não há fatores dominantes a aumentar o risco neste caso."]

    # simulação de threshold
    threshold_curve = []
    for t in [i / 100 for i in range(5, 96, 5)]:
        if proba >= max(0.7, t):
            dec = "REJECT"
        elif proba >= t:
            dec = "REVIEW"
        else:
            dec = "APPROVE"
        threshold_curve.append({"threshold": float(t), "decision": dec})

    # proxy de tendência de devolução (estimativa)
    tenure = None
    for col in ["loan_tenure_months", "num__loan_tenure_months"]:
        if col in store.feature_cols:
            try:
                tenure = float(X[col].iloc[0])
                break
            except Exception:
                tenure = None

    if tenure is None:
        tenure = 12.0
    tenure = max(6.0, min(60.0, tenure))

    hazard_base = 0.03 + 0.25 * proba
    trend = []
    for m in range(1, int(tenure) + 1):
        p = 1.0 - float(np.exp(-hazard_base * (m / tenure)))
        trend.append({"month": m, "estimated_default_probability": p})

    charts = {
        "local_bar": [
            {"name": (it.get("display_feature") or it["feature"]), "value": float(it["shap_value"])}
            for it in items[:10]
        ],
        "waterfall": wf,  # { base_value, contributions[] }
        "population_hist": [{"score": float(s)} for s in pop_scores.tolist()[:3000]],
        "population_marker": {"score": proba, "percentile": percentile},
        "threshold_curve": threshold_curve,
        "repayment_trend_proxy": {
            "note": "Estimativa (proxy). Não é histórico real de pagamentos.",
            "tenure_months": float(tenure),
            "series": trend,
        },
        "reason_codes": reason_codes[:10],
    }

    return {
        "application_id": application_id,
        "risk_probability": float(proba),
        "decision": decision,
        "threshold": float(threshold),
        "avg_population_score": pop_avg,
        "std_population_score": pop_std,
        "percentile_vs_population": percentile,
        "narrative": narrative,
        "key_drivers": drivers,
        "what_to_improve": improve,
        "charts": charts,
    }