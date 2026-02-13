import React, { useEffect, useState } from "react";
import { getDashboard } from "../api/endpoints";

export function Dashboard() {
  const [data, setData] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function load() {
    setBusy(true);
    setErr(null);
    try {
      const d = await getDashboard(500);
      setData(d);
    } catch (e: any) {
      setErr(e?.message ?? "Erro ao carregar dashboard");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div style={{ color: "#e5e7eb" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h2 style={{ margin: 0, color: "#ffffff" }}>Dashboard Overview</h2>
          <span style={{ fontSize: 12, opacity: 0.75 }}>
            Portfolio snapshot • Risk distribution • Drift monitoring
          </span>
        </div>

        <button onClick={load} disabled={busy} style={btn}>
          {busy ? "A atualizar..." : "Refresh"}
        </button>
      </div>

      <div style={{ height: 12 }} />

      {err && <div style={alert("high")}>{err}</div>}

      {!data ? (
        <div style={{ opacity: 0.8 }}>No datas yet…</div>
      ) : (
        <>
          <div style={grid}>
            <KpiCard
              title="Total Applications"
              value={String(data.total_applications)}
              sub="Amostra analisada"
            />
            <KpiCard
              title="High Risk (≥ 0.7)"
              value={String(data.high_risk)}
              sub="Requer revisão"
              tone="high"
            />
            <KpiCard
              title="Medium Risk (0.3–0.7)"
              value={String(data.medium_risk)}
              sub="Análise recomendada"
              tone="medium"
            />
            <KpiCard
              title="Low Risk (< 0.3)"
              value={String(data.low_risk)}
              sub="Aprovável"
              tone="low"
            />
            <KpiCard
              title="Avg Risk Score"
              value={Number(data.avg_risk_score).toFixed(3)}
              sub="Média da população"
            />
            <KpiCard
              title="Std Risk Score"
              value={Number(data.std_risk_score).toFixed(3)}
              sub="Dispersão do risco"
            />
          </div>

          <div style={{ height: 12 }} />

          <div style={alert(data.score_drift_detected ? data.score_drift_severity : "low")}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
              <div>
                <b>Score Drift:</b>{" "}
                {data.score_drift_detected ? "DETECTED" : "Not detected"}{" "}
                <span style={{ opacity: 0.8 }}>
                  — Severity {String(data.score_drift_severity).toUpperCase()}
                </span>
              </div>
              <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 12, opacity: 0.85 }}>
                PSI {Number(data.score_drift_psi).toFixed(3)} | KS {Number(data.score_drift_ks).toFixed(3)}
              </div>
            </div>

            <div style={{ height: 10 }} />

            {/* mini-barra visual do PSI só para dar “cara de sistema” */}
            <div style={psiBarWrap}>
              <div
                style={{
                  ...psiBarFill,
                  width: `${Math.max(2, Math.min(100, Number(data.score_drift_psi) * 100))}%`,
                }}
              />
            </div>
            <div style={{ fontSize: 12, opacity: 0.8, marginTop: 6 }}>
            PSI is an indicator of change in the score distribution. (monitorization).
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function KpiCard(props: { title: string; value: string; sub?: string; tone?: "low" | "medium" | "high" }) {
  const tone = props.tone ?? "low";
  return (
    <div style={card}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
        <div style={{ fontSize: 12, opacity: 0.8, color: "#334155" }}>{props.title}</div>
        <span style={pill(tone)}>{tone.toUpperCase()}</span>
      </div>

      <div style={{ fontSize: 28, fontWeight: 900, color: "#0f172a", marginTop: 6 }}>
        {props.value}
      </div>

      {props.sub && (
        <div style={{ fontSize: 12, opacity: 0.75, color: "#475569", marginTop: 2 }}>{props.sub}</div>
      )}
    </div>
  );
}

const grid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
  gap: 12,
};

const card: React.CSSProperties = {
  borderRadius: 16,
  padding: 14,
  background: "rgba(255,255,255,0.92)",
  border: "1px solid rgba(255,255,255,0.35)",
  boxShadow: "0 10px 30px rgba(0,0,0,.22)",
  backdropFilter: "blur(8px)",
};

const btn: React.CSSProperties = {
  padding: "8px 12px",
  borderRadius: 12,
  border: "1px solid rgba(2,6,23,.18)",
  background: "rgba(255,255,255,0.95)",
  color: "#0f172a",          // <- texto escuro
  fontWeight: 800,
  cursor: "pointer",
  boxShadow: "0 10px 24px rgba(0,0,0,.12)",
};

function pill(tone: "low" | "medium" | "high"): React.CSSProperties {
  const bg =
    tone === "high" ? "rgba(239,68,68,.12)" : tone === "medium" ? "rgba(245,158,11,.14)" : "rgba(34,197,94,.12)";
  const fg =
    tone === "high" ? "#b91c1c" : tone === "medium" ? "#92400e" : "#166534";
  const border =
    tone === "high" ? "rgba(239,68,68,.25)" : tone === "medium" ? "rgba(245,158,11,.25)" : "rgba(34,197,94,.25)";
  return {
    fontSize: 11,
    padding: "4px 8px",
    borderRadius: 999,
    background: bg,
    color: fg,
    border: `1px solid ${border}`,
    fontWeight: 700,
    whiteSpace: "nowrap",
  };
}

function alert(sev: string): React.CSSProperties {
  const s = String(sev).toLowerCase();

  // fundo claro (legível), mesmo no tema dark global
  const border =
    s === "high"
      ? "1px solid rgba(239,68,68,.30)"
      : s === "medium"
      ? "1px solid rgba(245,158,11,.30)"
      : "1px solid rgba(34,197,94,.30)";

  const bg =
    s === "high"
      ? "rgba(255, 241, 242, 0.92)"
      : s === "medium"
      ? "rgba(255, 247, 237, 0.92)"
      : "rgba(240, 253, 244, 0.92)";

  return {
    border,
    background: bg,
    borderRadius: 16,
    padding: 14,
    color: "#0f172a",
    boxShadow: "0 10px 30px rgba(0,0,0,.18)",
    backdropFilter: "blur(8px)",
  };
}

const psiBarWrap: React.CSSProperties = {
  height: 10,
  borderRadius: 999,
  background: "rgba(15,23,42,0.10)",
  overflow: "hidden",
};

const psiBarFill: React.CSSProperties = {
  height: "100%",
  borderRadius: 999,
  background: "linear-gradient(90deg, rgba(34,197,94,.85), rgba(245,158,11,.85), rgba(239,68,68,.85))",
};