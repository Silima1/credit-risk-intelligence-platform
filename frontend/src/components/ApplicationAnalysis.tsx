import React, { useEffect, useMemo, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid } from "recharts";
import { listApplications, predictById, xaiLocalById } from "../api/endpoints";

export function ApplicationAnalysis() {
  const [apps, setApps] = useState<any[]>([]);
  const [appId, setAppId] = useState<any>("");
  const [threshold, setThreshold] = useState(0.5);

  const [pred, setPred] = useState<any>(null);
  const [xai, setXai] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    listApplications(100)
      .then((r) => {
        setApps(r.items);
        setAppId(r.items?.[0] ?? "");
      })
      .catch((e) => setErr(e?.message ?? "Erro ao carregar aplicações"));
  }, []);

  async function analyze() {
    setBusy(true);
    setErr(null);
    setPred(null);
    setXai(null);
    try {
      const p = await predictById(appId, threshold);
      const lx = await xaiLocalById(appId);
      setPred(p);
      setXai(lx);
    } catch (e: any) {
      setErr(e?.message ?? "Erro ao analisar pedido");
    } finally {
      setBusy(false);
    }
  }

  const chartData = useMemo(() => {
    if (!xai?.items) return [];
    return xai.items.map((it: any) => ({
      name: it.display_feature ?? it.feature,
      value: Number(it.shap_value),
      impact: it.impact_text ?? "",
    }));
  }, [xai]);

  return (
    <div style={{ color: "#e5e7eb" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
        <div>
          <h2 style={{ margin: 0, color: "#fff" }}>Credit Application Analysis</h2>
          <div style={{ fontSize: 12, opacity: 0.8 }}>
            Per-application decision • Local XAI • Narrative drivers
          </div>
        </div>
        <button onClick={analyze} disabled={busy || !appId} style={btnPrimary}>
          {busy ? "A analisar..." : "Analyze"}
        </button>
      </div>

      <div style={{ height: 12 }} />

      {err && <div style={alertBox("high")}>{err}</div>}

      {/* Surface (claro) para ficar sempre legível em tema dark */}
      <div style={surface}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, alignItems: "end" }}>
          <div>
            <label style={label}>Application ID</label>
            <select value={appId} onChange={(e) => setAppId(e.target.value)} style={input}>
              {apps.map((id) => (
                <option key={String(id)} value={id}>
                  {String(id)}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={label}>Threshold</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              style={input}
            />
          </div>
        </div>

        <div style={{ height: 14 }} />

        {pred && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
              <Card title="Risk Probability" value={Number(pred.risk_probability).toFixed(3)} sub={`${Math.round(pred.risk_probability * 100)}% de risco estimado`} />
              <Card title="Prediction" value={pred.prediction === 1 ? "HIGH RISK" : "LOW RISK"} />
              <Card title="Recommendation" value={pred.recommendation} tone={pred.recommendation === "APPROVE" ? "low" : pred.recommendation === "REVIEW" ? "medium" : "high"} />
            </div>

            <div style={{ height: 12 }} />

            {/* mini barra de risco */}
            <div style={{ marginTop: 4 }}>
              <div style={{ fontSize: 12, color: "#475569", marginBottom: 6 }}>Risk gauge</div>
              <div style={gaugeWrap}>
                <div style={{ ...gaugeFill, width: `${Math.max(0, Math.min(100, pred.risk_probability * 100))}%` }} />
              </div>
              <div style={{ fontSize: 12, color: "#64748b", marginTop: 6 }}>
                0.0 (baixo) → 1.0 (alto) • Threshold atual: {Number(pred.used_threshold ?? threshold).toFixed(2)}
              </div>
            </div>
          </>
        )}

        {xai && (
          <>
            <div style={{ height: 18 }} />
            <h3 style={{ margin: 0, color: "#0f172a" }}>Key factors influencing the decision</h3>
            <div style={{ fontSize: 12, color: "#64748b", marginTop: 6 }}>
              Right bar = higher risk • Left bar = lower risk
            </div>

            <div style={{ height: 12 }} />

            <div style={panel}>
              <div style={{ height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={chartData}
                    layout="vertical"
                    margin={{ top: 10, right: 18, bottom: 10, left: 26 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <ReferenceLine x={0} />
                    <XAxis type="number" tick={{ fill: "#0f172a", fontSize: 12 }} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={220}
                      tick={{ fill: "#0f172a", fontSize: 12 }}
                    />
                    <Tooltip content={<XaiTooltip />} />
                    <Bar dataKey="value" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div style={{ height: 12 }} />

            <div style={panel}>
              <h3 style={{ margin: "0 0 10px", color: "#0f172a" }}>Factor → Impact on risk</h3>
              <table style={table}>
                <thead>
                  <tr>
                    <th style={th}>Factor</th>
                    <th style={th}>Impact on risk</th>
                  </tr>
                </thead>
                <tbody>
                  {xai.items.map((it: any) => (
                    <tr key={it.feature}>
                      <td style={td}>{it.display_feature ?? it.feature}</td>
                      <td style={td}>{it.impact_text ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ height: 12 }} />

            <div style={panel}>
              <h3 style={{ margin: "0 0 10px", color: "#0f172a" }}>Resume (To make decision)</h3>
              <ol style={{ margin: 0, paddingLeft: 18, color: "#0f172a" }}>
                {(xai.reason_codes ?? []).slice(0, 6).map((r: string, i: number) => (
                  <li key={i} style={{ marginBottom: 6 }}>
                    {r}
                  </li>
                ))}
              </ol>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function XaiTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const p = payload[0]?.payload;
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.98)",
        border: "1px solid rgba(15,23,42,.15)",
        borderRadius: 12,
        padding: 10,
        boxShadow: "0 18px 40px rgba(0,0,0,.20)",
        color: "#0f172a",
        maxWidth: 360,
      }}
    >
      <div style={{ fontWeight: 800, marginBottom: 6 }}>{p?.name}</div>
      <div style={{ fontSize: 12, opacity: 0.85 }}>
        SHAP: <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>{Number(p?.value).toFixed(6)}</span>
      </div>
      {p?.impact ? <div style={{ marginTop: 6, fontSize: 12 }}>{p.impact}</div> : null}
    </div>
  );
}

function Card(props: { title: string; value: string; sub?: string; tone?: "low" | "medium" | "high" }) {
  return (
    <div style={card}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
        <div style={{ fontSize: 12, opacity: 0.8, color: "#334155" }}>{props.title}</div>
        {props.tone ? <span style={pill(props.tone)}>{props.tone.toUpperCase()}</span> : null}
      </div>
      <div style={{ fontSize: 26, fontWeight: 900, color: "#0f172a", marginTop: 6 }}>{props.value}</div>
      {props.sub ? <div style={{ fontSize: 12, color: "#64748b", marginTop: 2 }}>{props.sub}</div> : null}
    </div>
  );
}

/* ================== styles (dark-safe) ================== */

const surface: React.CSSProperties = {
  marginTop: 10,
  borderRadius: 18,
  padding: 16,
  background: "rgba(255,255,255,0.92)",
  border: "1px solid rgba(255,255,255,0.35)",
  boxShadow: "0 16px 50px rgba(0,0,0,.22)",
  backdropFilter: "blur(10px)",
};

const panel: React.CSSProperties = {
  borderRadius: 16,
  padding: 14,
  background: "rgba(255,255,255,0.96)",
  border: "1px solid rgba(15,23,42,.12)",
  boxShadow: "0 12px 34px rgba(0,0,0,.14)",
};

const label: React.CSSProperties = { display: "block", fontSize: 12, color: "#334155", marginBottom: 6, fontWeight: 700 };

const input: React.CSSProperties = {
  width: "100%",
  padding: 10,
  borderRadius: 12,
  border: "1px solid rgba(15,23,42,.18)",
  background: "rgba(255,255,255,0.98)",
  color: "#0f172a",
  outline: "none",
};

const card: React.CSSProperties = {
  borderRadius: 16,
  padding: 14,
  background: "rgba(255,255,255,0.92)",
  border: "1px solid rgba(15,23,42,.12)",
  boxShadow: "0 12px 34px rgba(0,0,0,.12)",
};

const table: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  color: "#0f172a",
};

const th: React.CSSProperties = {
  textAlign: "left",
  borderBottom: "1px solid rgba(15,23,42,.10)",
  padding: 10,
  fontSize: 12,
  color: "#334155",
};

const td: React.CSSProperties = {
  borderBottom: "1px solid rgba(15,23,42,.06)",
  padding: 10,
  fontSize: 13,
  color: "#0f172a",
};

const btnPrimary: React.CSSProperties = {
  padding: "10px 14px",
  borderRadius: 12,
  border: "1px solid rgba(255,255,255,0.18)",
  background: "rgba(15,23,42,0.55)",
  color: "#fff",
  cursor: "pointer",
  boxShadow: "0 10px 26px rgba(0,0,0,.25)",
};

function alertBox(sev: "low" | "medium" | "high"): React.CSSProperties {
  const border =
    sev === "high" ? "1px solid rgba(239,68,68,.30)" : sev === "medium" ? "1px solid rgba(245,158,11,.30)" : "1px solid rgba(34,197,94,.30)";
  const bg =
    sev === "high" ? "rgba(255, 241, 242, 0.92)" : sev === "medium" ? "rgba(255, 247, 237, 0.92)" : "rgba(240, 253, 244, 0.92)";
  return {
    border,
    background: bg,
    borderRadius: 16,
    padding: 12,
    color: "#0f172a",
    boxShadow: "0 12px 34px rgba(0,0,0,.12)",
  };
}

function pill(tone: "low" | "medium" | "high"): React.CSSProperties {
  const bg =
    tone === "high" ? "rgba(239,68,68,.12)" : tone === "medium" ? "rgba(245,158,11,.14)" : "rgba(34,197,94,.12)";
  const fg = tone === "high" ? "#b91c1c" : tone === "medium" ? "#92400e" : "#166534";
  const border =
    tone === "high" ? "rgba(239,68,68,.25)" : tone === "medium" ? "rgba(245,158,11,.25)" : "rgba(34,197,94,.25)";
  return {
    fontSize: 11,
    padding: "4px 8px",
    borderRadius: 999,
    background: bg,
    color: fg,
    border: `1px solid ${border}`,
    fontWeight: 800,
    whiteSpace: "nowrap",
  };
}

const gaugeWrap: React.CSSProperties = {
  height: 10,
  borderRadius: 999,
  background: "rgba(15,23,42,0.10)",
  overflow: "hidden",
};

const gaugeFill: React.CSSProperties = {
  height: "100%",
  borderRadius: 999,
  background: "linear-gradient(90deg, rgba(34,197,94,.85), rgba(245,158,11,.85), rgba(239,68,68,.85))",
};