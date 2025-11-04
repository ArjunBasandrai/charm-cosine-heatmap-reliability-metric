import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const form = await req.formData();
  const flaskBase = process.env.FLASK_BASE_URL ?? "http://127.0.0.1:5000";

  const resp = await fetch(`${flaskBase}/process`, {
    method: "POST",
    body: form, // forwards the multipart form-data
  });

  const contentType = resp.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const json = await resp.json();
    return NextResponse.json(json, { status: resp.status });
  }

  const blob = await resp.blob();
  return new Response(blob, { status: resp.status, headers: { "content-type": contentType } });
}
