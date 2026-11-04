// DEMO MODE ONLY — sample values to preview the UI. Never used when VITE_DEMO_MODE=false.
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function upload() { await sleep(300); return { image_id: 'demo-image' }; }

export async function infer(device) {
  const cpu = device === 'cpu';
  await sleep(cpu ? 4200 : 3000);
  return {
    predicted_class: 'nv', confidence: 0.91, status: 'Completed (demo)',
    execution_time_ms: cpu ? 182 : 24,
    utilization_percent: cpu ? 78 : 61,
    gpu_memory_mb: cpu ? 0 : 412,
    images_per_second: cpu ? 5.5 : 41.7,
  };
}

export async function results() { await sleep(200); return {}; }
export async function metrics() {
  await sleep(300);
  return { cpu: { execution_time_ms: 182, images_per_second: 5.5 }, gpu: { execution_time_ms: 24, images_per_second: 41.7 } };
}
export async function system() {
  await sleep(300);
  return { cpu: { utilization_percent: 78 }, gpu: { utilization_percent: 61, memory_mb: 412 } };
}
export async function history() {
  await sleep(300);
  const mk = (shift) => Array.from({ length: 15 }, (_, i) => {
    const e = i + 1;
    return {
      epoch: e,
      train_acc: +(0.62 + 0.2 * (1 - Math.exp(-e / 5)) + shift).toFixed(3),
      val_acc: +(0.6 + 0.17 * (1 - Math.exp(-e / 5)) + shift).toFixed(3),
      train_loss: +(1.2 * Math.exp(-e / 6) + 0.45 - shift).toFixed(3),
      val_loss: +(1.25 * Math.exp(-e / 6) + 0.5 - shift).toFixed(3),
    };
  });
  return { cpu: mk(0), gpu: mk(0.004) };
}
