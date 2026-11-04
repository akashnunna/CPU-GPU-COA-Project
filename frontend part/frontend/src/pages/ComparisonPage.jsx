import { useCallback, useEffect, useState } from 'react';
import UploadZone from '../components/UploadZone.jsx';
import RunCard from '../components/RunCard.jsx';
import ComparisonSection from '../components/ComparisonSection.jsx';
import { uploadImage } from '../services/api.js';

const CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'];

export default function ComparisonPage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageId, setImageId] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState({});
  const [resetKey, setResetKey] = useState(0);

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const clear = () => { setFile(null); setPreviewUrl(null); setImageId(null); setResults({}); setResetKey((k) => k + 1); };

  const onSelect = async (f) => {
    clear();
    setError(null);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setUploading(true);
    try { setImageId((await uploadImage(f)).image_id); }
    catch (e) { setError(`Upload failed: ${e.message}`); }
    finally { setUploading(false); }
  };

  const onDone = useCallback((device, result) => setResults((r) => ({ ...r, [device]: result })), []);

  return (
    <div className="space-y-8">
      <section className="text-center">
        <h1 className="text-3xl font-bold sm:text-4xl">CPU vs GPU Performance</h1>
        <p className="mx-auto mt-2 max-w-2xl text-slate-500">
          A Computer Organization and Architecture project comparing CPU and GPU deep learning inference on the HAM10000 skin lesion dataset using PyTorch.
        </p>
      </section>

      <UploadZone file={file} previewUrl={previewUrl} uploading={uploading} onSelect={onSelect} onRemove={clear} />
      {error && <div className="rounded-xl bg-rose-50 p-3 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">{error}</div>}

      <div className="grid gap-6 md:grid-cols-2" key={resetKey}>
        <RunCard device="cpu" imageId={imageId} previewUrl={previewUrl} onDone={onDone} />
        <RunCard device="gpu" imageId={imageId} previewUrl={previewUrl} onDone={onDone} />
      </div>

      {results.cpu && results.gpu && <ComparisonSection cpu={results.cpu} gpu={results.gpu} />}

      <section className="card text-sm text-slate-600 dark:text-slate-300">
        <h3 className="mb-2 font-semibold">Dataset and model information</h3>
        <p>HAM10000: 10,015 dermatoscopic images in 7 classes ({CLASSES.join(', ')}). Lesion-aware split: ~7,000 train / ~1,000 validation / ~2,000 test. The same trained model, weights, preprocessing and input image are used for CPU and GPU runs.</p>
        <p className="mt-2 font-medium">Academic performance comparison only — not a medical diagnosis application.</p>
      </section>
    </div>
  );
}
