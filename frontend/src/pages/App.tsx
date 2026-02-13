// frontend/src/App.tsx
import React, { useEffect, useState } from "react";
import { getMeta } from "../api/endpoints";
import { Tabs } from "../components/Tabs";
import { Dashboard } from "../components/Dashboard";
import { ApplicationAnalysis } from "../components/ApplicationAnalysis";
import { XaiPage } from "../components/XaiPage";
import { OptimizePage } from "../components/OptimizePage";
import { DriftPage } from "../components/DriftPage";

// ✅ Coloca a tua imagem aqui: frontend/src/assets/hero-fintech.png
import hero from "../assets/hero-fintech.png";

export default function App() {
  const [features, setFeatures] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getMeta()
      .then((m) => setFeatures(m.feature_cols))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          color: "rgba(255,255,255,.9)",
          background:
            "radial-gradient(1200px 900px at 70% 0%, #0b2a64 0%, transparent 65%)," +
            "radial-gradient(900px 700px at 0% 10%, #122b56 0%, transparent 60%)," +
            "linear-gradient(180deg, #050b17, #071127)",
        }}
      >
        <div
          style={{
            padding: 18,
            borderRadius: 18,
            border: "1px solid rgba(255,255,255,.14)",
            background: "rgba(255,255,255,.08)",
            backdropFilter: "blur(10px)",
            boxShadow: "0 18px 60px rgba(0,0,0,.35)",
          }}
        >
          A carregar…
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        color: "rgba(255,255,255,.92)",
        fontFamily: "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
        background:
          "radial-gradient(1200px 900px at 70% 0%, #0b2a64 0%, transparent 65%)," +
          "radial-gradient(900px 700px at 0% 10%, #122b56 0%, transparent 60%)," +
          "linear-gradient(180deg, #050b17, #071127)",
      }}
    >
      {/* HERO */}
      <div style={{ position: "relative", height: 240, overflow: "hidden" }}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundImage: `linear-gradient(90deg, rgba(5,11,23,.92), rgba(5,11,23,.65)), url(${hero})`,
            backgroundSize: "cover",
            backgroundPosition: "center",
            filter: "saturate(1.15) contrast(1.05)",
            transform: "scale(1.02)",
          }}
        />

        <div style={{ position: "relative", height: "100%" }}>
          <div style={{ width: "min(1200px, calc(100% - 32px))", margin: "0 auto", paddingTop: 34 }}>
            <div
              style={{
                padding: 18,
                borderRadius: 24,
                border: "1px solid rgba(255,255,255,.14)",
                background: "rgba(255,255,255,.08)",
                backdropFilter: "blur(10px)",
                boxShadow: "0 18px 60px rgba(0,0,0,.35)",
              }}
            >
              <div style={{ display: "flex", gap: 12, justifyContent: "space-between", flexWrap: "wrap" }}>
                <div>
                  <div style={{ fontSize: 28, fontWeight: 800, margin: 0 }}>Credit Risk Intelligence Platform</div>
                  <div style={{ fontSize: 12, opacity: 0.75, marginTop: 6 }}>
                    ABI - TASK 2 | Intelligent Credit Risk, XAI & Monitoring
                  </div>
                </div>

                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  <span
                    style={{
                      padding: "6px 10px",
                      borderRadius: 999,
                      border: "1px solid rgba(255,255,255,.14)",
                      background: "rgba(255,255,255,.06)",
                      fontSize: 12,
                      opacity: 0.85,
                    }}
                  >
                    Features: {features.length}
                  </span>
                  <span
                    style={{
                      padding: "6px 10px",
                      borderRadius: 999,
                      border: "1px solid rgba(255,255,255,.14)",
                      background: "rgba(255,255,255,.06)",
                      fontSize: 12,
                      opacity: 0.85,
                    }}
                  >
                    API: OK
                  </span>
                </div>
              </div>

              <div style={{ marginTop: 12, height: 1, background: "rgba(255,255,255,.10)" }} />

              <div style={{ marginTop: 14, opacity: 0.75, fontSize: 12 }}>
                Real-time dashboards • Explainable decisions • Monitoring (drift) • Threshold optimization
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* PAGE */}
      <div style={{ width: "min(1200px, calc(100% - 32px))", margin: "0 auto", marginTop: 18, paddingBottom: 40 }}>
        <div
          style={{
            padding: 18,
            borderRadius: 24,
            border: "1px solid rgba(255,255,255,.14)",
            background: "rgba(255,255,255,.08)",
            backdropFilter: "blur(10px)",
            boxShadow: "0 18px 60px rgba(0,0,0,.35)",
          }}
        >
          <Tabs
            tabs={[
              { key: "dash", title: "Dashboard Overview", node: <Dashboard /> },
              { key: "app", title: "Credit Application Analysis", node: <ApplicationAnalysis /> },
              { key: "xai", title: "XAI Explanations", node: <XaiPage /> },
              { key: "opt", title: "Optimization", node: <OptimizePage /> },
              { key: "drift", title: "Adaptation / Drift", node: <DriftPage /> },
            ]}
          />
        </div>

        <div style={{ marginTop: 14, fontSize: 12, opacity: 0.6, textAlign: "center" }}>
          © Leonel Silima — manangement prototype - 2026.
        </div>
      </div>
    </div>
  );
}