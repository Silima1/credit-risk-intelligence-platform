// frontend/src/api/endpoints.ts
import { api } from "./client";

/* =========================
   META / HEALTH
========================= */

export async function getMeta() {
  const { data } = await api.get("/meta");
  return data as {
    feature_cols: string[];
    has_application_id: boolean;
  };
}

export async function healthCheck() {
  const { data } = await api.get("/health");
  return data as { status: string };
}

/* =========================
   DASHBOARD
========================= */

export async function getDashboard(sample_size = 500) {
  const { data } = await api.get("/dashboard", {
    params: { sample_size },
  });
  return data as {
    total_applications: number;
    high_risk: number;
    medium_risk: number;
    low_risk: number;
    avg_risk_score: number;
    std_risk_score: number;
    score_drift_detected: boolean;
    score_drift_severity: string;
    score_drift_psi: number;
    score_drift_ks: number;
  };
}

/* =========================
   APPLICATIONS
========================= */

export async function listApplications(limit = 100) {
  const { data } = await api.get("/applications", {
    params: { limit },
  });
  return data as { items: any[] };
}

/* =========================
   PREDICTION
========================= */

export async function predictById(application_id: any, threshold = 0.5) {
  const { data } = await api.post("/predict", {
    application_id,
    threshold,
  });

  return data as {
    risk_probability: number;
    prediction: number;
    recommendation: string;
    used_threshold: number;
  };
}

/* =========================
   XAI
========================= */

export async function xaiLocalById(application_id: any) {
  const { data } = await api.post("/xai/local", {
    application_id,
    threshold: 0.5,
  });

  return data as {
    items: {
      feature: string;
      display_feature?: string;
      shap_value: number;
      abs_shap: number;
      impact_text?: string;
    }[];
    reason_codes: string[];
  };
}

export async function xaiGlobal(sample_size = 200) {
  const { data } = await api.get("/xai/global", {
    params: { sample_size },
  });

  return data as {
    items: {
      feature: string;
      display_feature?: string;
      mean_abs_shap: number;
      std_shap: number;
    }[];
  };
}

/* =========================
   OPTIMIZATION
========================= */

export async function optimizeThreshold(cost_matrix: {
  FP: number;
  FN: number;
  TP: number;
  TN: number;
}) {
  const { data } = await api.post("/optimize/threshold", {
    cost_matrix,
  });

  return data as {
    optimal_threshold: number;
    expected_cost: number;
  };
}

/* =========================
   DRIFT
========================= */

export async function driftScore(sample_size = 200) {
  const { data } = await api.get("/drift/score", {
    params: { sample_size },
  });

  return data as {
    psi: number;
    ks: number;
    drift_detected: boolean;
    severity: string;
  };
}

/* =========================
   REPORT (XAI por cliente)
========================= */

export async function getReport(application_id: any, threshold = 0.5) {
  const { data } = await api.get(`/report/${application_id}`, {
    params: { threshold },
  });

  return data as {
    application_id: string;
    risk_probability: number;
    decision: "APPROVE" | "REVIEW" | "REJECT";
    threshold: number;
    percentile_vs_population: number;
    key_drivers: {
      title: string;
      direction: "increases" | "decreases";
      strength: "small" | "medium" | "large";
      shap_value: number;
      narrative: string;
    }[];
    what_to_improve: string[];
    charts: {
      local_bar: { name: string; value: number }[];
      population_hist: { score: number }[];
      threshold_curve: { threshold: number; decision: number }[];
    };
  };
}