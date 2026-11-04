import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';
import { motion } from 'framer-motion';
import ChartCard from './ChartCard.jsx';

export default function ComparisonSection({ cpu, gpu }) {
  const speedup = cpu.execution_time_ms / gpu.execution_time_ms;
  const ips = (r) => r.images_per_second ?? 1000 / r.execution_time_ms;
  const cards = [
    ['CPU execution time', `${cpu.execution_time_ms} ms`, 'text-cpu'],
    ['GPU execution time', `${gpu.execution_time_ms} ms`, 'text-gpu'],
    ['Speedup', `${speedup.toFixed(2)}×`, ''],
    ['CPU utilization', `${cpu.utilization_percent ?? '—'}%`, 'text-cpu'],
    ['GPU utilization', `${gpu.utilization_percent ?? '—'}%`, 'text-gpu'],
    ['GPU memory usage', `${gpu.gpu_memory_mb ?? '—'} MB`, 'text-gpu'],
    ['CPU images / sec', ips(cpu).toFixed(1), 'text-cpu'],
    ['GPU images / sec', ips(gpu).toFixed(1), 'text-gpu'],
  ];
  const time = [{ name: 'Time (ms)', CPU: cpu.execution_time_ms, GPU: gpu.execution_time_ms }];
  const thr = [{ name: 'Images/sec', CPU: +ips(cpu).toFixed(2), GPU: +ips(gpu).toFixed(2) }];
  const util = [{ name: 'Utilization %', CPU: cpu.utilization_percent ?? 0, GPU: gpu.utilization_percent ?? 0 }];
  const chart = (data) => (
    <BarChart data={data}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="name" /><YAxis /><Tooltip /><Legend />
      <Bar dataKey="CPU" fill="#3b82f6" radius={[6, 6, 0, 0]} /><Bar dataKey="GPU" fill="#10b981" radius={[6, 6, 0, 0]} />
    </BarChart>
  );
  return (
    <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
      <h2 className="text-xl font-bold">CPU vs GPU Comparison</h2>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {cards.map(([k, v, c]) => (
          <div key={k} className="card !p-4"><div className="text-xs text-slate-500">{k}</div><div className={`mt-1 text-2xl font-bold ${c}`}>{v}</div></div>
        ))}
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <ChartCard title="Execution time" height={220}>{chart(time)}</ChartCard>
        <ChartCard title="Throughput" height={220}>{chart(thr)}</ChartCard>
        <ChartCard title="Utilization" height={220}>{chart(util)}</ChartCard>
      </div>
    </motion.section>
  );
}
