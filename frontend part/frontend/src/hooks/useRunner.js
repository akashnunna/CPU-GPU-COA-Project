import { useCallback, useEffect, useRef, useState } from 'react';
import { runInference } from '../services/api.js';

export const STAGES = [
  'Image loading', 'Image preprocessing', 'Model initialization',
  'Forward propagation', 'Classification', 'Prediction completed',
];

const initial = { status: 'idle', stage: -1, progress: 0, result: null, error: null };

// Stage progress is estimated on the client while the request runs;
// the final stage completes only when the backend responds.
export default function useRunner(device, onDone) {
  const [s, setS] = useState(initial);
  const timer = useRef(null);
  useEffect(() => () => clearInterval(timer.current), []);

  const run = useCallback(async (imageId) => {
    clearInterval(timer.current);
    setS({ status: 'running', stage: 0, progress: 5, result: null, error: null });
    let st = 0;
    timer.current = setInterval(() => {
      st = Math.min(st + 1, 4);
      setS((p) => (p.status === 'running' ? { ...p, stage: st, progress: Math.round(((st + 0.5) / 6) * 100) } : p));
    }, 800);
    try {
      const result = await runInference(device, imageId);
      clearInterval(timer.current);
      setS({ status: 'done', stage: 6, progress: 100, result, error: null });
      onDone?.(device, result);
    } catch (e) {
      clearInterval(timer.current);
      setS((p) => ({ ...p, status: 'error', error: e.message }));
    }
  }, [device, onDone]);

  return { ...s, run };
}
