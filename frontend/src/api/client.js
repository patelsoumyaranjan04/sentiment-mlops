// frontend/src/api/client.js
// Thin wrapper around the FastAPI backend

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const predictSingle = (review) =>
  apiFetch("/predict", {
    method: "POST",
    body: JSON.stringify({ review }),
  });

export const predictBatch = (reviews) =>
  apiFetch("/predict/batch", {
    method: "POST",
    body: JSON.stringify({ reviews }),
  });

export const checkHealth = () => apiFetch("/health");
export const checkReady  = () => apiFetch("/ready");
