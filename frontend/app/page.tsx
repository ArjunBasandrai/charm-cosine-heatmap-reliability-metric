"use client";

import { useState } from "react";

type MapsResponse = {
  T: string;
  I: string;
  F: string;
  G: string;
  R: number;
};

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MapsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    if (!file) {
      setError("Select an image first.");
      return;
    }
    const form = new FormData();
    form.append("image", file);
    setLoading(true);
    try {
      const resp = await fetch("/api/process", { method: "POST", body: form });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      // Support either {T,I,F,gradcam,R} or {outputs:{...}} if your Flask returns nested data.
      const payload = data.outputs ? data.outputs : data;

      // Expect base64 PNG strings for T,I,F,gradcam and numeric R
      const maps: MapsResponse = {
        T: payload.T,
        I: payload.I,
        F: payload.F,
        G: payload.G,
        R: payload.R,
      };
      setResult(maps);
    } catch (err: any) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 900, margin: "40px auto", padding: 16, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 24, marginBottom: 16 }}>Run CHARM + Grad-CAM</h1>

      <form onSubmit={onSubmit} style={{ display: "grid", gap: 12 }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button
          type="submit"
          disabled={loading || !file}
          style={{
            padding: "10px 14px",
            borderRadius: 8,
            border: "1px solid #ccc",
            background: loading ? "#ddd" : "#f5f5f5",
            cursor: loading || !file ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Processing…" : "Process"}
        </button>
      </form>

      {error && (
        <p style={{ color: "crimson", marginTop: 12, whiteSpace: "pre-wrap" }}>{error}</p>
      )}

      {result && (
        <section style={{ marginTop: 24 }}>
          <div style={{ marginBottom: 12, fontSize: 16 }}>
            <strong>R:</strong> {Number.isFinite(result.R) ? result.R.toFixed(4) : String(result.R)}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: 12,
              alignItems: "start",
            }}
          >
            <Figure title="T (reliable)">
              <img
                alt="T map"
                style={{ width: "100%", height: "auto", borderRadius: 8, border: "1px solid #eee" }}
                src={`data:image/png;base64,${result.T}`}
              />
            </Figure>

            <Figure title="I (ambiguity)">
              <img
                alt="I map"
                style={{ width: "100%", height: "auto", borderRadius: 8, border: "1px solid #eee" }}
                src={`data:image/png;base64,${result.I}`}
              />
            </Figure>

            <Figure title="F (noise)">
              <img
                alt="F map"
                style={{ width: "100%", height: "auto", borderRadius: 8, border: "1px solid #eee" }}
                src={`data:image/png;base64,${result.F}`}
              />
            </Figure>

            <Figure title="Grad-CAM">
              <img
                alt="Grad-CAM map"
                style={{ width: "100%", height: "auto", borderRadius: 8, border: "1px solid #eee" }}
                src={`data:image/png;base64,${result.G}`}
              />
            </Figure>
          </div>
        </section>
      )}
    </main>
  );
}

function Figure({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <figure style={{ margin: 0 }}>
      {children}
      <figcaption style={{ marginTop: 8, fontSize: 14, color: "#555" }}>{title}</figcaption>
    </figure>
  );
}
