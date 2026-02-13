import React from "react";

export function Card({
  children,
  style = {},
}: {
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.95)",
        color: "#0f172a",
        borderRadius: 16,
        padding: 16,
        boxShadow: "0 8px 24px rgba(0,0,0,.18)",
        ...style,
      }}
    >
      {children}
    </div>
  );
}