import React, { useState } from "react";
import { driftScore } from "../api/endpoints";

export function DriftPage() {
  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    try {
      const r = await driftScore(200);
      setResult(r);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Adaptation / Drift</h2>
      <button onClick={run} disabled={busy} style={{ padding: "10px 14px", borderRadius: 10, cursor: "pointer" }}>
        {busy ? "A analisar..." : "Run Score Drift Check"}
      </button>

      <div style={{ height: 12 }} />

      {result && (
        <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
          <div><b>PSI:</b> {result.psi.toFixed(4)}</div>
          <div><b>KS:</b> {result.ks.toFixed(4)}</div>
          <div><b>Drift:</b> {result.drift_detected ? "YES" : "NO"}</div>
          <div><b>Severity:</b> {String(result.severity).toUpperCase()}</div>
        </div>
      )}
    </div>
  );
}