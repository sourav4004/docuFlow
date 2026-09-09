'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import api from '@/lib/api';

interface ExecSummary {
  items: Array<{
    id: string;
    execution_type: string;
    task_type: string;
    status: string;
    priority: string;
    latency_ms: number | null;
    actual_cost: number;
    created_at: string;
  }>;
}

interface EventSummary {
  PENDING: number;
  PROCESSED: number;
  FAILED: number;
  DEAD: number;
}

interface ReviewSummary {
  pending: number;
  overdue: number;
  by_status: Record<string, number>;
}

interface MemorySummary {
  total: number;
  by_scope_type: Array<{ scope: string; type: string; count: number }>;
}

interface TimelineItem {
  event_type: string;
  occurred_at: string;
  severity: string;
  title: string;
}

interface CostSummary {
  total_cost_usd: number;
  total_tokens: number;
  execution_count: number;
  by_model: Record<string, number>;
}

function statCard(label: string, value: string | number, accent: string) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 shadow-sm">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
      <p className={`mt-1 text-2xl font-bold ${accent}`}>{value}</p>
    </div>
  );
}

const severityColor: Record<string, string> = {
  CRITICAL: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
  HIGH: 'bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300',
  MEDIUM: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
  LOW: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
  INFO: 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300',
};

export default function IntelligencePage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  const [workspaceId, setWorkspaceId] = useState<number | null>(null);
  const [executions, setExecutions] = useState<ExecSummary['items']>([]);
  const [events, setEvents] = useState<EventSummary | null>(null);
  const [reviews, setReviews] = useState<ReviewSummary | null>(null);
  const [memory, setMemory] = useState<MemorySummary | null>(null);
  const [timeline, setTimeline] = useState<TimelineItem[]>([]);
  const [costs, setCosts] = useState<CostSummary | null>(null);
  const [snapshotName, setSnapshotName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const loadAll = useCallback(async () => {
    if (!workspaceId) return;
    setRefreshing(true);
    setError(null);
    try {
      const [execData, eventData, reviewData, memoryData, timelineData, costData] = await Promise.all([
        api.listExecutions(workspaceId, undefined),
        api.getEventSummary(workspaceId),
        api.getReviewSummary(workspaceId),
        api.getMemorySummary(workspaceId),
        api.getTimeline(workspaceId),
        api.getCostSummary(workspaceId),
      ]);
      setExecutions(execData.items || []);
      setEvents(eventData);
      setReviews(reviewData);
      setMemory(memoryData);
      setTimeline(timelineData.items || []);
      setCosts(costData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load Knowledge OS');
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  }, [workspaceId]);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

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

  useEffect(() => {
    if (workspaceId) {
      loadAll();
    }
  }, [workspaceId, loadAll]);

  const handleQueueDemoExecution = async () => {
    if (!workspaceId) return;
    try {
      await api.createExecution(workspaceId, 'rag', 'knowledge_check');
      setNotice('Execution queued');
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to queue execution');
    }
  };

  const handleSnapshot = async () => {
    if (!workspaceId) return;
    try {
      await api.createSnapshot(workspaceId, snapshotName || `snapshot-${new Date().toISOString().slice(0, 10)}`);
      setNotice('Knowledge snapshot captured');
      setSnapshotName('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to capture snapshot');
    }
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center">
        <div className="animate-spin h-6 w-6 border-3 border-blue-600 border-t-transparent rounded-full" />
      </div>
    );
  }
  if (!user) return null;

  const running = executions.filter((e) => ['QUEUED', 'RUNNING', 'RETRYING', 'WAITING_APPROVAL', 'WAITING_TOOL'].includes(e.status)).length;
  const failed = executions.filter((e) => ['FAILED', 'TIMED_OUT'].includes(e.status)).length;

  return (
    <main className="min-h-screen bg-slate-50 dark:bg-slate-900">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <header className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Knowledge OS</h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Execution durability, event outbox, review queue, memory, and timeline
            </p>
          </div>
          <div className="flex gap-2">
            <a
              href="/ai"
              className="px-3 py-1.5 rounded-lg text-sm font-medium text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-blue-400"
            >
              AI Dashboard
            </a>
            <button
              onClick={handleQueueDemoExecution}
              className="px-3 py-1.5 rounded-lg text-sm font-medium bg-blue-600 text-white hover:bg-blue-700"
            >
              Queue demo execution
            </button>
            <button
              onClick={loadAll}
              disabled={refreshing}
              className="px-3 py-1.5 rounded-lg text-sm font-medium border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200 hover:border-blue-400"
            >
              {refreshing ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
        </header>

        {error && (
          <div role="alert" className="mb-4 px-4 py-3 rounded-lg bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm">
            {error}
          </div>
        )}
        {notice && (
          <div role="status" className="mb-4 px-4 py-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300 text-sm">
            {notice}
          </div>
        )}

        {loading ? (
          <div className="animate-pulse space-y-4">
            <div className="h-24 bg-slate-200 dark:bg-slate-700 rounded-xl" />
            <div className="h-64 bg-slate-200 dark:bg-slate-700 rounded-xl" />
          </div>
        ) : (
          <>
            <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8" aria-label="Knowledge OS summary">
              {statCard('Running / queued executions', running, 'text-blue-600 dark:text-blue-400')}
              {statCard('Failed executions', failed, 'text-red-600 dark:text-red-400')}
              {statCard('Pending reviews', reviews?.pending ?? 0, 'text-amber-600 dark:text-amber-400')}
              {statCard('Dead-letter events', events?.DEAD ?? 0, failed ? 'text-red-600 dark:text-red-400' : 'text-slate-600 dark:text-slate-300')}
              {statCard('Memories stored', memory?.total ?? 0, 'text-violet-600 dark:text-violet-400')}
              {statCard('Outbox pending', events?.PENDING ?? 0, 'text-sky-600 dark:text-sky-400')}
              {statCard('AI executions', executions.length, 'text-slate-900 dark:text-white')}
              {statCard('30d AI cost ($)', costs ? costs.total_cost_usd.toFixed(4) : '0.0000', 'text-emerald-600 dark:text-emerald-400')}
            </section>

            <div className="grid md:grid-cols-2 gap-6">
              <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Knowledge timeline</h2>
                  <span className="text-xs text-slate-500">{timeline.length} events</span>
                </div>
                {timeline.length === 0 ? (
                  <p className="text-sm text-slate-500 dark:text-slate-400 py-8 text-center">
                    No knowledge events yet. Upload and process a document to see the system react.
                  </p>
                ) : (
                  <ul className="space-y-3 max-h-96 overflow-y-auto" aria-label="Timeline entries">
                    {timeline.map((item, idx) => (
                      <li key={`${item.occurred_at}-${idx}`} className="flex items-start gap-3">
                        <span className={`shrink-0 px-2 py-0.5 rounded text-xs font-medium ${severityColor[item.severity] ?? severityColor.INFO}`}>
                          {item.severity}
                        </span>
                        <div className="min-w-0">
                          <p className="text-sm text-slate-800 dark:text-slate-100 truncate">{item.title}</p>
                          <p className="text-xs text-slate-500">
                            {item.event_type} · {new Date(item.occurred_at).toLocaleString()}
                          </p>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
                <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">Latest executions</h2>
                {executions.length === 0 ? (
                  <p className="text-sm text-slate-500 dark:text-slate-400 py-8 text-center">
                    No AI executions recorded. Queue a demo execution above.
                  </p>
                ) : (
                  <ul className="space-y-3 max-h-96 overflow-y-auto" aria-label="Executions">
                    {executions.slice(0, 15).map((e) => (
                      <li key={e.id} className="flex items-center justify-between text-sm">
                        <div className="min-w-0">
                          <p className="text-slate-800 dark:text-slate-100 truncate">
                            {e.execution_type} / {e.task_type}
                          </p>
                          <p className="text-xs text-slate-500">{e.priority} · {new Date(e.created_at).toLocaleString()}</p>
                        </div>
                        <span className={`shrink-0 px-2 py-0.5 rounded text-xs font-medium ${severityColor[e.status] ?? severityColor.INFO}`}>
                          {e.status}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </div>

            <section className="mt-6 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">Governance snapshot</h2>
              <div className="flex flex-wrap items-end gap-3">
                <div>
                  <label htmlFor="snapshot-name" className="block text-xs font-medium text-slate-500 mb-1">
                    Snapshot name
                  </label>
                  <input
                    id="snapshot-name"
                    value={snapshotName}
                    onChange={(e) => setSnapshotName(e.target.value)}
                    placeholder="e.g. monthly-board-review"
                    className="rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-3 py-2 text-sm text-slate-900 dark:text-white"
                  />
                </div>
                <button
                  onClick={handleSnapshot}
                  className="px-3 py-2 rounded-lg text-sm font-medium bg-slate-800 dark:bg-slate-100 text-white dark:text-slate-900 hover:opacity-90"
                >
                  Capture reproducible snapshot
                </button>
              </div>
              <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
                Snapshots capture workspace health, deadlines, conflicts, changes, and temporal facts at one point in time — never live state.
              </p>
            </section>
          </>
        )}
      </div>
    </main>
  );
}
