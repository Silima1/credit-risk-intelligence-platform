import React, { useMemo, useState } from "react";

type Tab = { key: string; title: string; node: React.ReactNode };

export function Tabs(props: { tabs: Tab[]; defaultKey?: string }) {
  const defaultKey = props.defaultKey ?? props.tabs?.[0]?.key ?? "";
  const [active, setActive] = useState(defaultKey);

  const activeTab = useMemo(() => {
    return props.tabs.find((t) => t.key === active) ?? props.tabs[0];
  }, [active, props.tabs]);

  return (
    <div>
      <div style={tabsWrap}>
        <div style={tabRow}>
          {props.tabs.map((t) => {
            const isActive = t.key === active;
            return (
              <button
                key={t.key}
                onClick={() => setActive(t.key)}
                style={{ ...tabBtn, ...(isActive ? tabBtnActive : tabBtnInactive) }}
              >
                {t.title}
              </button>
            );
          })}
        </div>
      </div>

      <div style={{ height: 12 }} />

      <div style={contentWrap}>{activeTab?.node}</div>
    </div>
  );
}

/* ================== styles (dark text) ================== */

const tabsWrap: React.CSSProperties = {
  borderRadius: 18,
  padding: 12,
  background: "rgba(255,255,255,0.10)",
  border: "1px solid rgba(255,255,255,0.18)",
  backdropFilter: "blur(10px)",
  boxShadow: "0 14px 40px rgba(0,0,0,.18)",
};

const tabRow: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 10,
  alignItems: "center",
};

const tabBtn: React.CSSProperties = {
  borderRadius: 12,
  padding: "10px 14px",
  cursor: "pointer",
  border: "1px solid rgba(15,23,42,.18)",
  fontWeight: 800,
  fontSize: 13,
  transition: "all 140ms ease",
  background: "rgba(255,255,255,0.92)",
  color: "#0f172a", // <- TEXTO ESCURO SEMPRE
};

const tabBtnActive: React.CSSProperties = {
  background: "rgba(255,255,255,0.98)",
  border: "1px solid rgba(2,6,23,.28)",
  color: "#0b2a6f", // azul carregado
  boxShadow: "0 12px 30px rgba(0,0,0,.18)",
};

const tabBtnInactive: React.CSSProperties = {
  background: "rgba(255,255,255,0.85)",
  border: "1px solid rgba(2,6,23,.16)",
  color: "#111827", // quase preto
};

const contentWrap: React.CSSProperties = {
  borderRadius: 18,
  padding: 16,
  background: "rgba(255,255,255,0.08)",
  border: "1px solid rgba(255,255,255,0.16)",
  backdropFilter: "blur(10px)",
};