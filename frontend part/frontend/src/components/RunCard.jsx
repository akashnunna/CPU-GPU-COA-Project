import { Cpu, Zap, Play, AlertTriangle } from 'lucide-react';
import useRunner from '../hooks/useRunner.js';
import StageList from './StageList.jsx';
import SimViz from './SimViz.jsx';
import ResultPanel from './ResultPanel.jsx';

const META = {
  cpu: { title: 'CPU', Icon: Cpu, text: 'text-cpu', btn: 'bg-cpu hover:bg-blue-500', bar: 'bg-cpu' },
  gpu: { title: 'GPU', Icon: Zap, text: 'text-gpu', btn: 'bg-gpu hover:bg-emerald-500', bar: 'bg-gpu' },
};

export default function RunCard({ device, imageId, previewUrl, onDone }) {
  const m = META[device];
  const r = useRunner(device, onDone);
  const label = { idle: 'Idle', running: 'Running…', done: 'Completed', error: 'Failed' }[r.status];

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <div className={`flex items-center gap-2 text-lg font-bold ${m.text}`}><m.Icon size={24} /> {m.title}</div>
        <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium dark:bg-slate-800">{label}</span>
      </div>
      <button disabled={!imageId || r.status === 'running'} onClick={() => r.run(imageId)} className={`btn w-full ${m.btn}`}>
        <Play size={16} /> Run on {m.title}
      </button>
      {!imageId && <p className="text-xs text-slate-500">Upload an image to enable execution.</p>}
      <div>
        <div className="mb-1 flex justify-between text-xs text-slate-500"><span>Progress</span><span>{r.progress}%</span></div>
        <div className="h-2.5 rounded-full bg-slate-200 dark:bg-slate-700"><div className={`h-2.5 rounded-full transition-all duration-500 ${m.bar}`} style={{ width: `${r.progress}%` }} /></div>
      </div>
      {r.status !== 'idle' && (
        <>
          <StageList stage={r.stage} status={r.status} />
          <p className="text-[11px] text-slate-500">Stage progress is estimated in the UI while the request runs.</p>
          <SimViz device={device} stage={r.stage} status={r.status} />
        </>
      )}
      {r.error && <div className="flex items-center gap-2 rounded-lg bg-rose-50 p-3 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-300"><AlertTriangle size={16} /> {r.error}</div>}
      {r.result && <ResultPanel result={r.result} previewUrl={previewUrl} />}
    </div>
  );
}
