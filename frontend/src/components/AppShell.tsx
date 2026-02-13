import React from "react";
import hero from "../assets/hero-fintech.png";

export function AppShell(props: {
  title: string;
  subtitle: string;
  activeTab: string;
  tabs: { key: string; label: string; onClick: () => void }[];
  children: React.ReactNode;
}) {
  return (
    <div>
      {/* HERO */}
      <div
        style={{
          position: "relative",
          height: 240,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundImage: `linear-gradient(90deg, rgba(5,11,23,.92), rgba(5,11,23,.65)), url(${hero})`,
            backgroundSize: "cover",
            backgroundPosition: "center",
            filter: "saturate(1.15) contrast(1.05)",
          }}
        />
        <div className="container" style={{ position: "relative", height: "100%", paddingTop: 34 }}>
          <div className="glass" style={{ padding: 18 }}>
            <div className="h1">{props.title}</div>
            <div className="small" style={{ marginTop: 6 }}>{props.subtitle}</div>

            <hr className="sep" />

            <div className="tabs">
              {props.tabs.map((t) => (
                <button
                  key={t.key}
                  className={`tab ${props.activeTab === t.key ? "active" : ""}`}
                  onClick={t.onClick}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* PAGE */}
      <div className="container" style={{ marginTop: 18, marginBottom: 40 }}>
        <div className="glass" style={{ padding: 18 }}>
          {props.children}
        </div>
      </div>
    </div>
  );
}