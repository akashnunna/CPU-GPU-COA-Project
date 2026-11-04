import { CheckCircle2, Circle, Loader2 } from 'lucide-react';
import { STAGES } from '../hooks/useRunner.js';

export default function StageList({ stage, status }) {
  return (
    <ol className="space-y-1.5">
      {STAGES.map((name, i) => {
        const done = i < stage;
        const current = i === stage && status === 'running';
        return (
          <li key={name} className={`flex items-center gap-2 text-sm ${done ? 'text-emerald-600 dark:text-emerald-400' : current ? 'font-semibold' : 'text-slate-400'}`}>
            {done ? <CheckCircle2 size={16} /> : current ? <Loader2 size={16} className="animate-spin" /> : <Circle size={16} />}
            {i + 1}. {name}
          </li>
        );
      })}
    </ol>
  );
}
