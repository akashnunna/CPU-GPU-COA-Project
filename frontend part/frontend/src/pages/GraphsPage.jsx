import { useEffect, useState } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';
import ChartCard from '../components/ChartCard.jsx';
import { getHistory, getMetrics, getSystem } from '../services/api.js';

const FILTERS = [['all', 'All graphs'], ['acc', 'Accuracy'], ['loss', 'Loss'], ['perf', 'Execution & throughput'], ['util', 'Utilization']];

const lines = (history, key) =>
  history.cpu.map((r, i) => ({ epoch: r.epoch, CPU: r[key], GPU: history.gpu[i]?.[key] }));

const Line2 = ({ data }) => (
  <LineChart data={data}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="epoch" /><YAxis domain={['auto', 'auto']} /><Tooltip /><Legend />
    <Line type="monotone" dataKey="CPU" stroke="#3b82f6" strokeWidth={2} dot={false} />
    <Line type="monotone" dataKey="GPU" stroke="#10b981" strokeWidth={2} dot={false} />
  </LineChart>
);
const Bar2 = ({ data }) => (
  <BarChart data={data}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="name" /><YAxis /><Tooltip /><Legend />
    <Bar dataKey="CPU" fill="#3b82f6" radius={[6, 6, 0, 0]} /><Bar dataKey="GPU" fill="#10b981" radius={[6, 6, 0, 0]} />
  </BarChart>
);

export default function GraphsPage() {
  const [filter, setFilter] = useState('all');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getHistory(), getMetrics(), getSystem()])
      .then(([history, metrics, system]) => setData({ history, metrics, system }))
      .catch((e) => setError(e.message));
  }, []);

  const show = (k) => filter === 'all' || filter === k;

  return (
    <div className="grid gap-6 md:grid-cols-[220px_1fr]">
      <aside className="card h-fit !p-3">
        <div className="mb-2 px-2 font-bold">Graphs</div>
        {FILTERS.map(([id, label]) => (
          <button key={id} onClick={() => setFilter(id)}
            className={`block w-full rounded-lg px-3 py-2 text-left text-sm ${filter === id ? 'bg-slate-900 text-white dark:bg-white dark:text-slate-900' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}>
            {label}
          </button>
        ))}
      </aside>
      <div className="space-y-4">
        {error && <div className="rounded-xl bg-rose-50 p-3 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">Could not load experiment data: {error}</div>}
        {!data && !error && <div className="card text-slate-500">Loading experiment data…</div>}
        {data && (
          <div className="grid gap-4 lg:grid-cols-2">
            {show('acc') && <ChartCard title="Training Accuracy"><Line2 data={lines(data.history, 'train_acc')} /></ChartCard>}
            {show('acc') && <ChartCard title="Validation Accuracy"><Line2 data={lines(data.history, 'val_acc')} /></ChartCard>}
            {show('loss') && <ChartCard title="Training Loss"><Line2 data={lines(data.history, 'train_loss')} /></ChartCard>}
            {show('loss') && <ChartCard title="Validation Loss"><Line2 data={lines(data.history, 'val_loss')} /></ChartCard>}
            {show('perf') && <ChartCard title="Execution Time (ms)"><Bar2 data={[{ name: 'Inference', CPU: data.metrics.cpu.execution_time_ms, GPU: data.metrics.gpu.execution_time_ms }]} /></ChartCard>}
            {show('perf') && <ChartCard title="Throughput (images/sec)"><Bar2 data={[{ name: 'Throughput', CPU: data.metrics.cpu.images_per_second, GPU: data.metrics.gpu.images_per_second }]} /></ChartCard>}
            {show('util') && <ChartCard title="Resource Utilization (%)"><Bar2 data={[{ name: 'Utilization', CPU: data.system.cpu.utilization_percent, GPU: data.system.gpu.utilization_percent }]} /></ChartCard>}
          </div>
        )}
      </div>
    </div>
  );
}
