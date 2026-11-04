import { useState } from 'react';
import Navbar from './components/Navbar.jsx';
import ComparisonPage from './pages/ComparisonPage.jsx';
import GraphsPage from './pages/GraphsPage.jsx';
import useTheme from './hooks/useTheme.js';
import { DEMO } from './services/api.js';

export default function App() {
  const [page, setPage] = useState('comparison');
  const { dark, toggle } = useTheme();
  return (
    <div className="min-h-screen">
      <Navbar page={page} setPage={setPage} dark={dark} toggleTheme={toggle} />
      {DEMO && (
        <div className="bg-amber-100 px-4 py-2 text-center text-xs font-medium text-amber-900 dark:bg-amber-900/40 dark:text-amber-200">
          DEMO MODE — all numbers are sample values for previewing the UI, not real experiment results. Set VITE_DEMO_MODE=false in .env to use the backend.
        </div>
      )}
      <main className="mx-auto max-w-6xl px-4 py-8">
        {page === 'comparison' ? <ComparisonPage /> : <GraphsPage />}
      </main>
    </div>
  );
}
