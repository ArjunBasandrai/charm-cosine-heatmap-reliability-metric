"use client";

import { useState, DragEvent } from "react";

type MapsResponse = {
  T: string;
  I: string;
  F: string;
  G: string;
  R: number;
  predClass: string;
};

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MapsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleFileSelect(f: File | null) {
    setFile(f);
    setResult(null);
    setError(null);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  }

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
      const payload = data.outputs ? data.outputs : data;
      setResult({
        T: payload.T,
        I: payload.I,
        F: payload.F,
        G: payload.G,
        R: payload.R,
        predClass: String(payload.class ?? ""),
      });
    } catch (err: any) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const rPct = result ? Math.min(100, Math.max(0, Number(result.R) * 100)) : 0;

  return (
    <main className="max-w-4xl mx-auto p-6 font-sans">
      <h1 className="text-3xl font-semibold mb-6 text-slate-100">CHARM + Grad-CAM Visualization</h1>

      <form onSubmit={onSubmit} className="flex flex-col gap-6">
        <div
          className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center cursor-pointer bg-slate-800 hover:bg-slate-700 transition"
          onDrop={onDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => document.getElementById("file-input")?.click()}
        >
          <p className="text-slate-300">{file ? `Selected: ${file.name}` : "Click or drag an image here"}</p>
          <input
            id="file-input"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => handleFileSelect(e.target.files?.[0] ?? null)}
          />
        </div>

        {file && (
          <img
            src={URL.createObjectURL(file)}
            alt="Selected preview"
            className="w-full max-h-[500px] object-contain rounded-xl border border-slate-700"
          />
        )}

        <button
          type="submit"
          disabled={loading || !file}
          className={`px-6 py-3 rounded-xl font-medium text-slate-900 transition ${
            loading || !file ? "bg-slate-500 cursor-not-allowed" : "bg-emerald-500 hover:bg-emerald-400"
          }`}
        >
          {loading ? "Processing…" : "Process"}
        </button>
      </form>

      {error && <p className="mt-4 text-red-400 whitespace-pre-wrap">{error}</p>}

      {result && (
        <section className="mt-10">
          <div className="grid gap-4 sm:grid-cols-2 items-center mb-6">
            <div className="flex items-center gap-3">
              <span className="text-slate-300 font-medium">Predicted class</span>
              <span className="inline-flex items-center rounded-full bg-emerald-600/20 text-emerald-300 border border-emerald-600/40 px-3 py-1 text-sm font-semibold">
                {result.predClass || "—"}
              </span>
            </div>

            <div>
              <div className="flex items-center justify-between">
                <span className="text-slate-300 font-medium">R score</span>
                <span className="text-slate-300">{rPct.toFixed(1)}%</span>
              </div>
              <div className="mt-2 h-3 w-full rounded-full bg-slate-700 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-700"
                  style={{ width: `${rPct}%` }}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Number(rPct.toFixed(1))}
                  role="progressbar"
                />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <Figure title="T (reliable)">
              <img src={`data:image/png;base64,${result.T}`} className="w-full h-auto rounded-lg border border-slate-700" alt="T map" />
            </Figure>
            <Figure title="I (ambiguity)">
              <img src={`data:image/png;base64,${result.I}`} className="w-full h-auto rounded-lg border border-slate-700" alt="I map" />
            </Figure>
            <Figure title="F (noise)">
              <img src={`data:image/png;base64,${result.F}`} className="w-full h-auto rounded-lg border border-slate-700" alt="F map" />
            </Figure>
            <Figure title="Grad-CAM">
              <img src={`data:image/png;base64,${result.G}`} className="w-full h-auto rounded-lg border border-slate-700" alt="Grad-CAM map" />
            </Figure>
          </div>
        </section>
      )}
    </main>
  );
}

function Figure({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <figure className="flex flex-col items-center">
      <div className="border border-slate-700 rounded-lg overflow-hidden shadow-md">{children}</div>
      <figcaption className="mt-2 text-sm text-slate-400">{title}</figcaption>
    </figure>
  );
}
