import React, { useState } from "react";
import { xaiGlobal } from "../api/endpoints";

export function XaiPage() {
  const [items, setItems] = useState<any[] | null>(null);
  const [busy, setBusy] = useState(false);

  async function load() {
    setBusy(true);
    try {
      const r = await xaiGlobal(200);
      setItems(r.items);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Explainable AI (XAI)</h2>

      <button onClick={load} disabled={busy} style={{ padding: "10px 14px", borderRadius: 10, cursor: "pointer" }}>
        {busy ? "A calcular..." : "Load Global Importance"}
      </button>

      <div style={{ height: 12 }} />

      {items && (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={th}>Feature</th>
              <th style={th}>Mean |SHAP|</th>
              <th style={th}>Std</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it) => (
              <tr key={it.feature}>
                <td style={td}>{it.feature}</td>
                <td style={td}>{it.mean_abs_shap.toFixed(6)}</td>
                <td style={td}>{it.std_shap.toFixed(6)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

const th: React.CSSProperties = { textAlign: "left", borderBottom: "1px solid #eee", padding: 8 };
const td: React.CSSProperties = { borderBottom: "1px solid #f3f3f3", padding: 8 };