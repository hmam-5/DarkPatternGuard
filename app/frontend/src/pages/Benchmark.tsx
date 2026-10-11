import { useEffect, useState } from "react";
import { BarChart3 } from "lucide-react";
import { PageTitle } from "../App";
import { getModels, type ModelMetadata } from "../lib/api";

export default function Benchmark() {
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getModels()
      .then(setModels)
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Unable to load benchmark"),
      );
  }, []);

  const hasSavedEvaluation = models.some((model) => model.version === "saved");

  return (
    <>
      <PageTitle icon={BarChart3} title="Benchmark leaderboard">
        Compare serving models against saved evaluation artifacts.
      </PageTitle>
      {error && <p className="mb-4 text-red-600">{error}</p>}
      {!error && !hasSavedEvaluation && (
        <p
          className="mb-4 rounded-lg border border-amber-300 bg-amber-50 p-4 text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100"
          role="status"
        >
          No real-dataset benchmark is available yet. Fallback predictions are
          heuristic only and must not be cited as model evaluation.
        </p>
      )}
      <div className="card overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b dark:border-slate-700">
              <th className="p-3">Model</th>
              <th className="p-3">Version</th>
              <th className="p-3">Macro-F1</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr
                className="border-b last:border-0 dark:border-slate-700"
                key={model.model}
              >
                <td className="p-3 font-semibold">{model.model}</td>
                <td className="p-3">{model.version}</td>
                <td className="p-3">
                  {model.macro_f1 == null ? "—" : model.macro_f1.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {hasSavedEvaluation && (
        <div className="mt-6 grid gap-5 md:grid-cols-2">
          <Figure
            title="Confusion matrix"
            src="http://localhost:8000/static/results/confusion_matrix.png"
          />
          <Figure
            title="Calibration"
            src="http://localhost:8000/static/results/calibration.png"
          />
        </div>
      )}
    </>
  );
}

function Figure({ title, src }: { title: string; src: string }) {
  return (
    <div className="card">
      <h2 className="mb-3 font-semibold">{title}</h2>
      <img
        src={src}
        alt={title}
        className="min-h-40 w-full rounded-lg bg-slate-100 object-contain dark:bg-slate-900"
        onError={(e) => {
          e.currentTarget.style.display = "none";
        }}
      />
    </div>
  );
}
