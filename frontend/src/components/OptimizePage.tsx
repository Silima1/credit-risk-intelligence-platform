import React, { useState } from "react";
import { optimizeThreshold } from "../api/endpoints";

export function OptimizePage() {
  const [FP, setFP] = useState(100);
  const [FN, setFN] = useState(500);
  const [TP, setTP] = useState(-200);
  const [TN, setTN] = useState(-50);

  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    setResult(null);
    try {
      const r = await optimizeThreshold({ FP, FN, TP, TN });
      setResult(r);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Optimization</h2>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <Field label="False Positive Cost" value={FP} onChange={setFP} />
        <Field label="False Negative Cost" value={FN} onChange={setFN} />
        <Field label="True Positive Reward" value={TP} onChange={setTP} />
        <Field label="True Negative Reward" value={TN} onChange={setTN} />
      </div>

      <div style={{ height: 12 }} />

      <button onClick={run} disabled={busy} style={{ padding: "10px 14px", borderRadius: 10, cursor: "pointer" }}>
        {busy ? "A otimizar..." : "Calculate Optimal Threshold"}
      </button>

      <div style={{ height: 12 }} />

      {result && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <Card title="Optimal Threshold" value={result.optimal_threshold.toFixed(3)} />
          <Card title="Expected Cost" value={result.expected_cost.toFixed(2)} />
        </div>
      )}
    </div>
  );
}

function Field(props: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div>
      <label>{props.label}</label>
      <input
        type="number"
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
        style={{ width: "100%", padding: 10, borderRadius: 10 }}
      />
    </div>
  );
}

function Card(props: { title: string; value: string }) {
  return (
    <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
      <div style={{ fontSize: 12, opacity: 0.7 }}>{props.title}</div>
      <div style={{ fontSize: 20, fontWeight: 700 }}>{props.value}</div>
    </div>
  );
}