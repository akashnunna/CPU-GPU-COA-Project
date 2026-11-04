# CPU vs GPU Performance — Frontend
React + Vite + Tailwind + Framer Motion + Recharts + Lucide.

    npm install
    npm run dev        (http://localhost:5173)

Edit `.env`: VITE_DEMO_MODE=true previews the UI with labelled sample values;
set false and VITE_API_URL to your FastAPI server to use real data.

Expected FastAPI endpoints (see src/services/api.js):
POST /upload, POST /infer/cpu, POST /infer/gpu, GET /results, GET /history, GET /metrics, GET /system
(enable CORS for http://localhost:5173 in FastAPI).
Academic performance comparison only — not a medical diagnosis tool.
