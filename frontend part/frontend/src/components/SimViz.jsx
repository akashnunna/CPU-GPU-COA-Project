import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';

// SIMULATED processing-state visualization. These are NOT real hardware registers.
const UNITS = {
  cpu: ['PC', 'IR', 'MAR', 'MDR', 'ACC', 'R1', 'R2', 'R3'],
  gpu: ['Scheduler', 'SM0', 'SM1', 'SM2', 'SM3', 'Shared Mem', 'Global Mem', 'Tensor Cores'],
};
const ACTIVE = {
  cpu: [['PC', 'MAR', 'MDR'], ['MDR', 'R1', 'R2'], ['MAR', 'MDR', 'IR'], ['ACC', 'R1', 'R2', 'R3'], ['ACC', 'R3'], ['PC']],
  gpu: [['Global Mem', 'Scheduler'], ['SM0', 'SM1', 'Shared Mem'], ['Scheduler', 'Global Mem'], ['SM0', 'SM1', 'SM2', 'SM3', 'Tensor Cores'], ['Tensor Cores', 'Shared Mem'], ['Scheduler']],
};
const hex = () => '0x' + Math.floor(Math.random() * 65535).toString(16).padStart(4, '0').toUpperCase();

export default function SimViz({ device, stage, status }) {
  const [vals, setVals] = useState({});
  const idx = status === 'done' ? 5 : Math.max(0, Math.min(stage, 5));
  const active = status === 'idle' ? [] : ACTIVE[device][idx];

  useEffect(() => {
    if (status !== 'running') return;
    const t = setInterval(() => setVals(Object.fromEntries(UNITS[device].map((u) => [u, hex()]))), 400);
    return () => clearInterval(t);
  }, [status, device]);

  const mem = status === 'idle' ? 0 : status === 'done' ? 100 : Math.round(((idx + 1) / 6) * 100);
  const color = device === 'cpu' ? 'bg-cpu' : 'bg-gpu';
  const ring = device === 'cpu' ? 'border-cpu' : 'border-gpu';

  return (
    <div className="space-y-3 rounded-xl bg-slate-50 p-3 dark:bg-slate-800/50">
      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Simulated processing states (not real hardware registers)</div>
      <div className="flex items-center justify-between gap-1 text-[11px] font-medium text-slate-500">
        <span>Data input</span><ArrowRight size={12} /><span>{device === 'cpu' ? 'Registers / ALU' : 'SMs / cores'}</span><ArrowRight size={12} /><span>Memory</span><ArrowRight size={12} /><span>Output</span>
      </div>
      <div className="grid grid-cols-4 gap-2">
        {UNITS[device].map((u) => {
          const on = active.includes(u);
          return (
            <motion.div key={u} animate={{ scale: on ? 1.05 : 1, opacity: on ? 1 : 0.5 }}
              className={`rounded-lg border-2 p-1.5 text-center ${on ? `${ring} bg-white dark:bg-slate-900` : 'border-transparent bg-slate-200 dark:bg-slate-800'}`}>
              <div className="truncate text-[10px] font-bold">{u}</div>
              <div className="font-mono text-[10px] text-slate-500">{on && status === 'running' ? vals[u] || '0x0000' : on ? 'active' : '—'}</div>
            </motion.div>
          );
        })}
      </div>
      <div>
        <div className="mb-1 flex justify-between text-xs text-slate-500"><span>Memory usage (simulated)</span><span>{mem}%</span></div>
        <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-700"><motion.div className={`h-2 rounded-full ${color}`} animate={{ width: `${mem}%` }} /></div>
      </div>
    </div>
  );
}
