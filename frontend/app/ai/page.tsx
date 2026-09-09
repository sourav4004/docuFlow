'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import api from '@/lib/api';

interface HealthData {
  total_documents: number;
  active_knowledge: number;
  stale_knowledge: number;
  processing_failures: number;
  metadata_completeness: number;
  duplicate_rate: number;
  conflict_rate: number;
  avg_health: number | null;
  ai_usage: { ai_executions_30d: number; ai_cost_30d: number; success_rate: number | null };
}

interface ActionSummary {
  status_counts: Record<string, number>;
  pending_actions: number;
  running: number;
  completed: number;
  failed: number;
}

interface Suggestion {
  id: number;
  title: string;
  description: string | null;
  reason: string | null;
  suggestion_type: string;
  priority: string;
  status: string;
  created_at: string;
}

interface DeadlineItem {
  id: number;
  title: string;
  due_date: string;
  status: string;
  confidence: string;
  source: string;
}

export default function AiDashboardPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  const [workspaceId, setWorkspaceId] = useState<number | null>(null);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [actionSummary, setActionSummary] = useState<ActionSummary | null>(null);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [deadlines, setDeadlines] = useState<DeadlineItem[]>([]);
  const [feedback, setFeedback] = useState<{ total_feedback: number; acceptance_rate: number | null; categories: Record<string, number> } | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Copilot
  const [copilotQuestion, setCopilotQuestion] = useState('');
  const [copilotAnswer, setCopilotAnswer] = useState<string | null>(null);
  const [copilotSources, setCopilotSources] = useState<number>(0);
  const [copilotLoading, setCopilotLoading] = useState(false);

  // Reports
  const [templates, setTemplates] = useState<Array<{ template: string; name: string }>>([]);
  const [reportResult, setReportResult] = useState<{ title: string; executive_summary: string; findings: Array<{ detail: string }> } | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  const loadAll = useCallback(async () => {
    if (!workspaceId) return;
    setRefreshing(true);
    setError(null);
    try {
      const [healthData, summary, suggestionData, deadlineData, feedbackData] = await Promise.all([
        api.getWorkspaceKnowledgeHealth(workspaceId),
        api.getActionCenterSummary(workspaceId),
        api.listSuggestions(workspaceId),
        api.listDeadlines(workspaceId),
        api.getFeedbackAnalytics(),
      ]);
      setHealth(healthData);
      setActionSummary(summary);
      setSuggestions((suggestionData.items || []).filter((s) => s.status === 'OPEN'));
      setDeadlines(deadlineData.items || []);
      setFeedback(feedbackData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load AI dashboard');
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  }, [workspaceId]);

  // Authentication check
  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  // Resolve workspace
  useEffect(() => {
    if (!user) return;
    api
      .listWorkspaces()
      .then((resp) => {
        const ws = resp.items?.[0];
        if (ws) setWorkspaceId(ws.id);
      })
      .catch(() => setError('Could not resolve workspace'));
  }, [user]);

  // Load report templates
  useEffect(() => {
    api
      .listReportTemplates()
      .then((resp) => setTemplates(resp.templates || []))
      .catch(() => undefined);
  }, []);

  // Initial load
  useEffect(() => {
    if (workspaceId) {
      loadAll();
    }
  }, [workspaceId, loadAll]);

  const handleGenerateSuggestions = async () => {
    if (!workspaceId) return;
    try {
      const resp = await api.generateSuggestions(workspaceId);
      await loadAll();
      setError(resp.suggestions_created > 0 ? null : 'No new suggestions to create');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate suggestions');
    }
  };

  const handleDismissSuggestion = async (id: number) => {
    try {
      await api.dismissSuggestion(id);
      setSuggestions((prev) => prev.filter((s) => s.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to dismiss suggestion');
    }
  };

  const handleCopilotAsk = async () => {
    if (!copilotQuestion.trim()) return;
    setCopilotLoading(true);
    setCopilotAnswer(null);
    setCopilotSources(0);
    try {
      const resp = await api.askCopilot(copilotQuestion.trim(), 'WORKSPACE');
      setCopilotAnswer(resp.answer);
      setCopilotSources(resp.sources?.length || 0);
    } catch (err) {
      setCopilotAnswer(err instanceof Error ? `Error: ${err.message}` : 'Copilot could not answer');
    } finally {
      setCopilotLoading(false);
    }
  };

  const handleGenerateReport = async (template: string) => {
    setReportLoading(true);
    setReportResult(null);
    try {
      const resp = await api.generateReport(template);
      setReportResult(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate report');
    } finally {
      setReportLoading(false);
    }
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center">
        <div className="flex items-center space-x-3 text-slate-600 dark:text-slate-300">
          <div className="animate-spin h-6 w-6 border-3 border-blue-600 border-t-transparent rounded-full" />
          <span className="font-medium text-sm">Loading DocuFlow AI...</span>
        </div>
      </div>
    );
  }

  if (!user) return null;

  const formatDate = (iso: string) => {
    try {
      return new Date(iso).toLocaleDateString();
    } catch {
      return iso;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-slate-100/70 to-slate-200/50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-800 text-slate-900 dark:text-slate-100">
      <header className="sticky top-0 z-30 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-xs font-bold text-lg">
                AI
              </div>
              <div>
                <span className="font-bold text-lg text-slate-900 dark:text-white tracking-tight">
                  DocuFlow AI
                </span>
                <span className="ml-2 px-2 py-0.5 text-2xs font-semibold uppercase tracking-wider bg-indigo-100 text-indigo-800 dark:bg-indigo-950 dark:text-indigo-300 rounded-full">
                  Knowledge Intelligence
                </span>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <a
                href="/dashboard"
                className="inline-flex items-center px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 hover:text-blue-600 border border-slate-200 dark:border-slate-700 rounded-lg transition-colors"
              >
                ← Workspace
              </a>
              <button
                onClick={loadAll}
                disabled={refreshing}
                className="inline-flex items-center px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 rounded-lg transition-colors disabled:opacity-50"
              >
                {refreshing ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {error && (
          <div className="p-4 rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-200 text-sm">
            {error}
            <button onClick={() => setError(null)} className="ml-3 font-medium hover:underline">
              Dismiss
            </button>
          </div>
        )}

        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {[0, 1, 2].map((i) => (
              <div key={i} className="bg-white dark:bg-slate-800/90 rounded-xl p-5 border border-slate-200/80 dark:border-slate-700 animate-pulse h-32" />
            ))}
          </div>
        ) : (
          <>
            {/* Knowledge Health */}
            <section aria-labelledby="health-heading">
              <h2 id="health-heading" className="text-lg font-bold tracking-tight text-slate-900 dark:text-white uppercase">
                Knowledge Health
              </h2>
              <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                {[
                  { label: 'Documents', value: health?.total_documents ?? 0 },
                  { label: 'Active', value: health?.active_knowledge ?? 0 },
                  { label: 'Processing Failures', value: health?.processing_failures ?? 0, alert: (health?.processing_failures ?? 0) > 0 },
                  { label: 'Duplicate Rate', value: health ? `${health.duplicate_rate}%` : '—' },
                  { label: 'Avg Health', value: health?.avg_health != null ? `${health.avg_health}/100` : '—' },
                  { label: 'AI Runs (30d)', value: health?.ai_usage.ai_executions_30d ?? 0 },
                ].map((card) => (
                  <div
                    key={card.label}
                    className={`bg-white dark:bg-slate-800/90 rounded-xl p-4 border ${
                      card.alert ? 'border-red-300 dark:border-red-800' : 'border-slate-200/80 dark:border-slate-700'
                    }`}
                  >
                    <p className="text-2xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                      {card.label}
                    </p>
                    <p className="mt-1 text-2xl font-bold text-slate-900 dark:text-white">{card.value}</p>
                  </div>
                ))}
              </div>
              {health && health.ai_usage.ai_cost_30d > 0 && (
                <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                  Estimated AI cost (30d): ${health.ai_usage.ai_cost_30d.toFixed(4)} · Success rate:{' '}
                  {health.ai_usage.success_rate != null ? `${Math.round(health.ai_usage.success_rate * 100)}%` : 'n/a'}
                </p>
              )}
            </section>

            {/* Action Center */}
            <section aria-labelledby="actions-heading">
              <div className="flex items-center justify-between">
                <h2 id="actions-heading" className="text-lg font-bold tracking-tight text-slate-900 dark:text-white uppercase">
                  AI Action Center
                </h2>
                <button
                  onClick={handleGenerateSuggestions}
                  className="inline-flex items-center px-3 py-1.5 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors"
                >
                  Generate Suggestions
                </button>
              </div>
              <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: 'Pending Approvals', value: actionSummary?.pending_actions ?? 0 },
                  { label: 'Running', value: actionSummary?.running ?? 0 },
                  { label: 'Completed', value: actionSummary?.completed ?? 0 },
                  { label: 'Failed', value: actionSummary?.failed ?? 0, alert: (actionSummary?.failed ?? 0) > 0 },
                ].map((card) => (
                  <div
                    key={card.label}
                    className={`bg-white dark:bg-slate-800/90 rounded-xl p-4 border ${
                      card.alert ? 'border-red-300 dark:border-red-800' : 'border-slate-200/80 dark:border-slate-700'
                    }`}
                  >
                    <p className="text-2xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">{card.label}</p>
                    <p className="mt-1 text-2xl font-bold text-slate-900 dark:text-white">{card.value}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Suggestions */}
            <section aria-labelledby="suggestions-heading">
              <h2 id="suggestions-heading" className="text-lg font-bold tracking-tight text-slate-900 dark:text-white uppercase">
                AI Suggestions
              </h2>
              {suggestions.length === 0 ? (
                <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">
                  No open suggestions. Suggestions are generated from observable events (document readiness, deadlines, duplicates, gaps) and never take autonomous action.
                </p>
              ) : (
                <ul className="mt-3 space-y-2">
                  {suggestions.map((s) => (
                    <li
                      key={s.id}
                      className="bg-white dark:bg-slate-800/90 rounded-xl p-4 border border-slate-200/80 dark:border-slate-700 flex items-start justify-between gap-3"
                    >
                      <div>
                        <p className="text-sm font-semibold text-slate-900 dark:text-white">{s.title}</p>
                        {s.reason && <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{s.reason}</p>}
                        <span className="mt-1 inline-block px-2 py-0.5 text-2xs font-semibold uppercase tracking-wider bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-300 rounded-full">
                          {s.suggestion_type}
                        </span>
                      </div>
                      <button
                        onClick={() => handleDismissSuggestion(s.id)}
                        className="text-xs font-medium text-slate-500 hover:text-slate-800 dark:hover:text-white shrink-0"
                        aria-label={`Dismiss suggestion: ${s.title}`}
                      >
                        Dismiss
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            {/* Copilot */}
            <section aria-labelledby="copilot-heading">
              <h2 id="copilot-heading" className="text-lg font-bold tracking-tight text-slate-900 dark:text-white uppercase">
                Workspace Copilot
              </h2>
              <div className="mt-3 bg-white dark:bg-slate-800/90 rounded-xl p-4 border border-slate-200/80 dark:border-slate-700">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={copilotQuestion}
                    onChange={(e) => setCopilotQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleCopilotAsk()}
                    placeholder="Ask about your workspace knowledge, e.g. 'Which contracts expire soon?'"
                    className="flex-1 px-3 py-2 text-sm bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                  <button
                    onClick={handleCopilotAsk}
                    disabled={copilotLoading || !copilotQuestion.trim()}
                    className="inline-flex items-center px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors disabled:opacity-50"
                  >
                    {copilotLoading ? 'Thinking...' : 'Ask'}
                  </button>
                </div>
                {copilotAnswer && (
                  <div className="mt-3 border-t border-slate-200 dark:border-slate-700 pt-3">
                    <p className="text-sm text-slate-800 dark:text-slate-200 whitespace-pre-wrap">{copilotAnswer}</p>
                    {copilotSources > 0 && (
                      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                        {copilotSources} source document(s) · grounded answer
                      </p>
                    )}
                  </div>
                )}
              </div>
            </section>

            {/* Deadlines */}
            <section aria-labelledby="deadlines-heading">
              <h2 id="deadlines-heading" className="text-lg font-bold tracking-tight text-slate-900 dark:text-white uppercase">
                Deadlines
              </h2>
              {deadlines.length === 0 ? (
                <p className="mt-3 text-sm text-slate-500 dark:text-slate-400">No deadlines tracked yet.</p>
              ) : (
                <ul className="mt-3 space-y-2">
                  {deadlines.map((d) => (
                    <li
                      key={d.id}
                      className="bg-white dark:bg-slate-800/90 rounded-xl p-4 border border-slate-200/80 dark:border-slate-700 flex items-center justify-between gap-3"
                    >
                      <div>
                        <p className="text-sm font-semibold text-slate-900 dark:text-white">{d.title}</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400">Due {formatDate(d.due_date)}</p>
                      </div>
                      <span
                        className={`px-2 py-0.5 text-2xs font-semibold uppercase tracking-wider rounded-full ${
                          d.status === 'OVERDUE'
                            ? 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300'
                            : d.status === 'DUE'
                            ? 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300'
                            : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
                        }`}
                      >
                        {d.status}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            {/* Feedback + Reports */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <section aria-labelledby="feedback-heading">
                <h2 id="feedback-heading" className="text-lg font-bold tracking-tight text-slate-900 dark:text-white uppercase">
                  AI Answer Feedback
                </h2>
                <div className="mt-3 bg-white dark:bg-slate-800/90 rounded-xl p-4 border border-slate-200/80 dark:border-slate-700">
                  <p className="text-sm text-slate-700 dark:text-slate-300">
                    {feedback && feedback.total_feedback > 0
                      ? `${feedback.total_feedback} ratings · ${Math.round((feedback.acceptance_rate ?? 0) * 100)}% acceptance`
                      : 'No feedback recorded yet.'}
                  </p>
                  {feedback && feedback.total_feedback > 0 && (
                    <ul className="mt-2 space-y-1">
                      {Object.entries(feedback.categories).map(([cat, count]) => (
                        <li key={cat} className="text-xs text-slate-500 dark:text-slate-400">
                          {cat}: {count}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </section>

              <section aria-labelledby="reports-heading">
                <h2 id="reports-heading" className="text-lg font-bold tracking-tight text-slate-900 dark:text-white uppercase">
                  AI Reports
                </h2>
                <div className="mt-3 bg-white dark:bg-slate-800/90 rounded-xl p-4 border border-slate-200/80 dark:border-slate-700 space-y-2">
                  {templates.map((t) => (
                    <button
                      key={t.template}
                      onClick={() => handleGenerateReport(t.template)}
                      disabled={reportLoading}
                      className="w-full text-left px-3 py-2 text-sm font-medium text-slate-700 dark:text-slate-200 bg-slate-50 dark:bg-slate-900 hover:bg-indigo-50 dark:hover:bg-indigo-950/40 border border-slate-200 dark:border-slate-700 rounded-lg transition-colors disabled:opacity-50"
                    >
                      {t.name}
                    </button>
                  ))}
                  {reportResult && (
                    <div className="pt-3 border-t border-slate-200 dark:border-slate-700">
                      <p className="text-sm font-semibold text-slate-900 dark:text-white">{reportResult.title}</p>
                      <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">{reportResult.executive_summary}</p>
                      {reportResult.findings.length > 0 && (
                        <ul className="mt-2 space-y-1">
                          {reportResult.findings.slice(0, 4).map((f, i) => (
                            <li key={i} className="text-xs text-slate-500 dark:text-slate-400">
                              • {f.detail}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
              </section>
            </div>
          </>
        )}
      </main>
    </div>
  );
}