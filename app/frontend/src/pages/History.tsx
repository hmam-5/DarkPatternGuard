import { useEffect, useState } from "react";
import { History as HistoryIcon, ThumbsDown, ThumbsUp } from "lucide-react";
import { PageTitle } from "../App";
import { getHistory, sendFeedback, type HistoryResponse, type Label } from "../lib/api";

export default function HistoryPage() {
  const [data, setData] = useState<HistoryResponse>(); const [error, setError] = useState("");
  useEffect(() => { getHistory().then(setData).catch((e) => setError(e instanceof Error ? e.message : "Unable to load history")); }, []);
  const vote = async (id: number, label: Label) => { try { await sendFeedback(id, label); } catch (e) { setError(e instanceof Error ? e.message : "Unable to save feedback"); } };
  return <><PageTitle icon={HistoryIcon} title="Prediction history">Review saved analyses and help improve the guard with feedback.</PageTitle>{error && <p className="mb-4 text-red-600">{error}</p>}<div className="card overflow-x-auto"><table className="w-full min-w-[650px] text-left text-sm"><thead><tr className="border-b dark:border-slate-700"><th className="p-3">Date</th><th className="p-3">Text</th><th className="p-3">Label</th><th className="p-3">Confidence</th><th className="p-3">Feedback</th></tr></thead><tbody>{data?.items.map((item) => <tr className="border-b last:border-0 dark:border-slate-700" key={item.id}><td className="p-3">{new Date(item.created_at).toLocaleDateString()}</td><td className="max-w-xs truncate p-3">{item.text ?? <span className="muted">Private text</span>}</td><td className="p-3 font-semibold">{item.final_label}</td><td className="p-3">{(item.confidence * 100).toFixed(1)}%</td><td className="p-3"><button aria-label="Agree with prediction" onClick={() => vote(item.id, item.final_label)} className="mr-2 rounded-lg p-2 hover:bg-mint/20"><ThumbsUp size={16} /></button><button aria-label="Disagree with prediction" onClick={() => vote(item.id, "Normal")} className="rounded-lg p-2 hover:bg-coral/20"><ThumbsDown size={16} /></button></td></tr>)}</tbody></table>{!data?.items.length && <p className="muted p-8 text-center">No predictions yet.</p>}</div></>;
}
