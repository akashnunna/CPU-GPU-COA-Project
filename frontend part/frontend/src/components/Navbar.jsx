import { Cpu, BarChart3, GitCompare, Moon, Sun } from 'lucide-react';

export default function Navbar({ page, setPage, dark, toggleTheme }) {
  const tab = (id, label, Icon) => (
    <button
      onClick={() => setPage(id)}
      className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition ${
        page === id
          ? 'bg-slate-900 text-white dark:bg-white dark:text-slate-900'
          : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800'
      }`}
    >
      <Icon size={16} /> {label}
    </button>
  );
  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-950/80">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-3">
        <div className="flex items-center gap-2 font-bold"><Cpu className="text-cpu" /> CPU vs GPU Performance</div>
        <nav className="flex items-center gap-2">
          {tab('comparison', 'Comparison', GitCompare)}
          {tab('graphs', 'Graphs', BarChart3)}
          <button onClick={toggleTheme} aria-label="Toggle theme" className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800">
            {dark ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </nav>
      </div>
    </header>
  );
}
