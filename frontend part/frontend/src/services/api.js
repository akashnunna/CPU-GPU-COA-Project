import * as demo from './demoData.js';

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const DEMO = import.meta.env.VITE_DEMO_MODE === 'true';

async function req(path, opts) {
  const res = await fetch(BASE + path, opts);
  if (!res.ok) throw new Error(`Request failed (${res.status}) ${path}`);
  return res.json();
}

// POST /upload (multipart "file") -> { image_id }
export function uploadImage(file) {
  if (DEMO) return demo.upload();
  const fd = new FormData();
  fd.append('file', file);
  return req('/upload', { method: 'POST', body: fd });
}

// POST /infer/{cpu|gpu} {image_id} ->
// { predicted_class, confidence(0-1), execution_time_ms, status, utilization_percent,
//   gpu_memory_mb (gpu only), images_per_second, output_image_url? }
export function runInference(device, imageId) {
  if (DEMO) return demo.infer(device);
  return req(`/infer/${device}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: imageId }),
  });
}

// GET /results -> latest stored run results
export const getResults = () => (DEMO ? demo.results() : req('/results'));
// GET /history -> { cpu: [{epoch, train_acc, val_acc, train_loss, val_loss}], gpu: [...] }
export const getHistory = () => (DEMO ? demo.history() : req('/history'));
// GET /metrics -> { cpu: {execution_time_ms, images_per_second}, gpu: {...} }
export const getMetrics = () => (DEMO ? demo.metrics() : req('/metrics'));
// GET /system -> { cpu: {utilization_percent}, gpu: {utilization_percent, memory_mb} }
export const getSystem = () => (DEMO ? demo.system() : req('/system'));
