import { motion } from 'framer-motion';

export default function ResultPanel({ result, previewUrl }) {
  const img = result.output_image_url || previewUrl;
  const rows = [
    ['Predicted class', result.predicted_class],
    ['Confidence', `${(result.confidence * 100).toFixed(1)}%`],
    ['Execution time', `${result.execution_time_ms} ms`],
    ['Status', result.status || 'Completed'],
  ];
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex gap-3 rounded-xl border border-slate-200 p-3 dark:border-slate-700">
      <img src={img} alt="Output" className="h-24 w-24 rounded-lg object-cover" />
      <dl className="flex-1 space-y-1 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2"><dt className="text-slate-500">{k}</dt><dd className="font-semibold">{v}</dd></div>
        ))}
      </dl>
    </motion.div>
  );
}
