import { useRef, useState } from 'react';
import { UploadCloud, X, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';

export default function UploadZone({ file, previewUrl, uploading, onSelect, onRemove }) {
  const input = useRef(null);
  const [drag, setDrag] = useState(false);

  const take = (f) => { if (f && f.type.startsWith('image/')) onSelect(f); };
  const onDrop = (e) => { e.preventDefault(); setDrag(false); take(e.dataTransfer.files?.[0]); };

  return (
    <div className="card">
      <input ref={input} type="file" accept="image/*" hidden onChange={(e) => { take(e.target.files?.[0]); e.target.value = ''; }} />
      {!file ? (
        <motion.div
          whileHover={{ scale: 1.005 }}
          onClick={() => input.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          className={`flex h-56 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed text-center transition ${
            drag ? 'border-cpu bg-blue-50 dark:bg-blue-950/30' : 'border-slate-300 dark:border-slate-700'
          }`}
        >
          <UploadCloud size={44} className="text-slate-400" />
          <div className="font-semibold">Upload an image</div>
          <div className="text-sm text-slate-500">Drag and drop a skin lesion image here, or click to browse</div>
        </motion.div>
      ) : (
        <div className="flex flex-col items-center gap-4 sm:flex-row">
          <img src={previewUrl} alt="Uploaded preview" className="h-56 w-full rounded-xl bg-slate-100 object-contain dark:bg-slate-800 sm:w-72" />
          <div className="flex-1 space-y-3 text-center sm:text-left">
            <div className="break-all font-semibold">{file.name}</div>
            <div className="text-sm text-slate-500">{(file.size / 1024).toFixed(1)} KB {uploading && '· uploading…'}</div>
            <div className="flex justify-center gap-2 sm:justify-start">
              <button onClick={() => input.current.click()} className="btn bg-slate-700 hover:bg-slate-600"><RefreshCw size={16} /> Replace</button>
              <button onClick={onRemove} className="btn bg-rose-600 hover:bg-rose-500"><X size={16} /> Remove</button>
            </div>
            <p className="text-xs text-slate-500">This same image is used for both CPU and GPU inference.</p>
          </div>
        </div>
      )}
    </div>
  );
}
