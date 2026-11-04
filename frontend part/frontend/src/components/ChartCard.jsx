import { ResponsiveContainer } from 'recharts';

export default function ChartCard({ title, children, height = 260 }) {
  return (
    <div className="card">
      <h3 className="mb-3 font-semibold">{title}</h3>
      <div style={{ height }}><ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer></div>
    </div>
  );
}
