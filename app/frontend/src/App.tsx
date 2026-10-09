import { useState, type ReactNode } from "react";
import { Link, NavLink, Route, Routes } from "react-router-dom";
import { BarChart3, History, Moon, ShieldCheck, Sun, Upload, Zap } from "lucide-react";
import Analyze from "./pages/Analyze";
import Compare from "./pages/Compare";
import HistoryPage from "./pages/History";
import Benchmark from "./pages/Benchmark";

export default function App() {
  const [dark, setDark] = useState(() => localStorage.getItem("dark-mode") === "true");
  const toggleDark = () => {
    setDark((value) => {
      localStorage.setItem("dark-mode", String(!value));
      return !value;
    });
  };
  return (
    <div className={dark ? "dark" : ""}>
      <div className="min-h-screen bg-slate-50 text-ink dark:bg-slate-950 dark:text-slate-100">
        <header className="border-b border-slate-200 bg-white/90 dark:border-slate-800 dark:bg-slate-900/90">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4">
            <Link to="/" className="flex items-center gap-2 text-lg font-bold"><ShieldCheck className="text-mint" /> DarkPatternGuard</Link>
            <nav className="hidden items-center gap-1 sm:flex">
              <NavLink className="rounded-lg px-3 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-800" to="/">Analyze</NavLink>
              <NavLink className="rounded-lg px-3 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-800" to="/compare">Compare</NavLink>
              <NavLink className="rounded-lg px-3 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-800" to="/history">History</NavLink>
              <NavLink className="rounded-lg px-3 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-800" to="/benchmark">Benchmark</NavLink>
            </nav>
            <button aria-label="Toggle dark mode" onClick={toggleDark} className="rounded-lg p-2 hover:bg-slate-100 dark:hover:bg-slate-800">{dark ? <Sun size={18} /> : <Moon size={18} />}</button>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-5 py-8"><Routes><Route path="/" element={<Analyze />} /><Route path="/compare" element={<Compare />} /><Route path="/history" element={<HistoryPage />} /><Route path="/benchmark" element={<Benchmark />} /></Routes></main>
      </div>
    </div>
  );
}

export function PageTitle({ icon: Icon, title, children }: { icon: typeof Zap; title: string; children?: ReactNode }) {
  return <div className="mb-7 flex items-start justify-between"><div className="flex gap-3"><Icon className="mt-1 text-coral" /><div><h1 className="text-3xl font-bold">{title}</h1><p className="muted mt-1">{children}</p></div></div></div>;
}

export const navIcons = { history: History, compare: Upload, benchmark: BarChart3 };
