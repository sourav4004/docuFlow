'use client';

import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import api from '@/lib/api';

type Dict = Record<string, unknown>;

const statusColor: Record<string, string> = {
  PROPOSED: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
  EVALUATING: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
  APPROVAL_REQUIRED: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
  APPROVED: 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300',
  STAGED: 'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300',
  ACTIVE: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  SUPERSEDED: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400',
  ROLLED_BACK: 'bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300',
  REJECTED: 'bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300',
  OPEN: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
  INVESTIGATING: 'bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300',
  MITIGATED: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
  RESOLVED: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  POSTMORTEM: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400',
  healthy: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  degraded: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
  critical: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
};

function badge(status: string | undefined) {
  const cls = statusColor[status ?? ''] ?? 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300';
  return (
    <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>
      {status ?? '—'}
    </span>
  );
}

function statCard(label: string, value: string | number, accent = 'text-slate-900 dark:text-white') {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 shadow-sm">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
      <p className={`mt-1 text-2xl font-bold ${accent}`}>{value}</p>
    </div>
  );
}

function panel(title: string, children: ReactNode, action?: ReactNode) {
  return (
    <section
      aria-label={title}
      className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm"
    >
      <div className="flex items-center justify-between gap-3 mb-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-300">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

function emptyRow(message = 'No data yet') {
  return (
    <p className="text-sm text-slate-500 dark:text-slate-400 py-4 text-center">{message}</p>
  );
}

export default function AiControlPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  const [workspaceId, setWorkspaceId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<Dict | null>(null);
  const [deps, setDeps] = useState<Dict | null>(null);
  const [improvements, setImprovements] = useState<Dict[]>([]);
  const [experiments, setExperiments] = useState<Dict[]>([]);
  const [incidents, setIncidents] = useState<Dict[]>([]);
  const [alerts, setAlerts] = useState<Dict[]>([]);
  const [alertSummary, setAlertSummary] = useState<Dict | null>(null);
  const [slo, setSlo] = useState<Dict[]>([]);
  const [scorecards, setScorecards] = useState<Dict[]>([]);
  const [knowledge, setKnowledge] = useState<Dict | null>(null);
  const [gaps, setGaps] = useState<Dict | null>(null);
  const [govAudit, setGovAudit] = useState<Dict[]>([]);
  const [safety, setSafety] = useState<Dict | null>(null);
  const [apiHealth, setApiHealth] = useState<Dict | null>(null);
  const [dbHealth, setDbHealth] = useState<Dict | null>(null);
  const [feedback, setFeedback] = useState<Dict | null>(null);
  const [agents, setAgents] = useState<Dict | null>(null);
  const [workflows, setWorkflows] = useState<Dict | null>(null);
  const [memory, setMemory] = useState<Dict | null>(null);
  const [graph, setGraph] = useState<Dict | null>(null);
  const [search, setSearch] = useState<Dict | null>(null);
  const [reports, setReports] = useState<Dict[]>([]);
  const [p23Providers, setP23Providers] = useState<Dict | null>(null);
  const [p23Vector, setP23Vector] = useState<Dict | null>(null);

  const loadAll = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const ws = workspaceId;
      const all = await Promise.allSettled([
        api.getSystemHealth({
          availability: 0.95, latency: 0.9, quality: 0.9,
          cost: 0.85, error_rate: 0.92, queue_health: 0.9,
          provider_health: 0.9,
        }),
        api.getDependencyGraph(),
        api.getImprovements(),
        api.getExperiments(),
        api.getIncidents(),
        api.getAlerts(),
        api.getAlertSummary(),
        api.getSloHistory(),
        ws ? api.getQualityScorecards(ws) : Promise.resolve([]),
        ws ? api.getKnowledgeHealth(ws) : Promise.resolve(null),
        ws ? api.getKnowledgeGaps(ws) : Promise.resolve({ gaps: [], recommendations: [] }),
        api.getGovernanceAudit(),
        api.getInjectionCorpus(),
        api.getApiHealth(),
        api.getDatabaseHealth(),
        ws ? api.getFeedbackSummary(ws) : Promise.resolve(null),
        ws ? api.getAgentIntelligence(ws) : Promise.resolve(null),
        ws ? api.getWorkflowIntelligence(ws) : Promise.resolve(null),
        ws ? api.getMemoryQuality(ws) : Promise.resolve(null),
        ws ? api.getGraphHealth(ws) : Promise.resolve(null),
        ws ? api.getSearchIntelligence(ws) : Promise.resolve(null),
        api.getReports(),
        api.getProviderReadiness(),
        api.getVectorActivation(),
      ]);
      const [
        h, d, imp, exp, inc, al, als, sl, sc, kh, kg, ga, sf, ah, dbh,
        fb, ag, wf, mem, gr, sr, rep, p23pr, p23ve,
      ] = all;
      if (h.status === 'fulfilled') setHealth(h.value);
      if (d.status === 'fulfilled') setDeps(d.value);
      if (imp.status === 'fulfilled') setImprovements(imp.value as Dict[]);
      if (exp.status === 'fulfilled') setExperiments(exp.value as Dict[]);
      if (inc.status === 'fulfilled') setIncidents(inc.value as Dict[]);
      if (al.status === 'fulfilled') setAlerts(al.value as Dict[]);
      if (als.status === 'fulfilled') setAlertSummary(als.value);
      if (sl.status === 'fulfilled') setSlo(sl.value as Dict[]);
      if (sc.status === 'fulfilled') setScorecards(sc.value as Dict[]);
      if (kh.status === 'fulfilled') setKnowledge(kh.value as Dict | null);
      if (kg.status === 'fulfilled') setGaps(kg.value);
      if (ga.status === 'fulfilled') setGovAudit(ga.value as Dict[]);
      if (sf.status === 'fulfilled') setSafety(sf.value);
      if (ah.status === 'fulfilled') setApiHealth(ah.value);
      if (dbh.status === 'fulfilled') setDbHealth(dbh.value);
      if (fb.status === 'fulfilled') setFeedback(fb.value);
      if (ag.status === 'fulfilled') setAgents(ag.value);
      if (wf.status === 'fulfilled') setWorkflows(wf.value);
      if (mem.status === 'fulfilled') setMemory(mem.value);
      if (gr.status === 'fulfilled') setGraph(gr.value);
      if (sr.status === 'fulfilled') setSearch(sr.value);
      if (rep.status === 'fulfilled') setReports(rep.value as Dict[]);
      if (p23pr.status === 'fulfilled') setP23Providers(p23pr.value as Dict | null);
      if (p23ve.status === 'fulfilled') setP23Vector(p23ve.value as Dict | null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load AI control center');
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

  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center">
        <div className="animate-spin h-6 w-6 border-3 border-blue-600 border-t-transparent rounded-full" />
      </div>
    );
  }
  if (!user) return null;

  const healthScore = typeof health?.health_score === 'number' ? health.health_score : null;
  const openIncidents = incidents.filter((i) => i.status === 'OPEN').length;
  const openAlerts = typeof alertSummary?.total_open === 'number' ? alertSummary.total_open : alerts.length;
  const activeImprovements = improvements.filter(
    (p) => p.status === 'ACTIVE' || p.status === 'STAGED',
  ).length;
  const detectionRate = typeof safety?.detection_rate === 'number'
    ? Math.round(Number(safety.detection_rate) * 100)
    : null;

  return (
    <main className="min-h-screen bg-slate-50 dark:bg-slate-900">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <header className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white">AI Control Center</h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Self-improvement, experiments, AI quality, knowledge health, governance, incidents, and safety
            </p>
          </div>
          <div className="flex gap-2">
            <a
              href="/ops"
              className="px-3 py-1.5 rounded-lg text-sm font-medium text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-blue-400"
            >
              Operations
            </a>
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
          <div className="mb-4 px-4 py-3 rounded-lg bg-amber-50 dark:bg-amber-950/50 border border-amber-200 dark:border-amber-800 text-sm text-amber-800 dark:text-amber-200">
            {error}
          </div>
        )}

        {loading ? (
          <div className="text-sm text-slate-500 dark:text-slate-400">Loading control center…</div>
        ) : (
          <div className="space-y-6">
            {/* Summary stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {statCard('System health', healthScore !== null ? `${Math.round(healthScore * 100)}%` : '—',
                healthScore !== null && healthScore >= 0.8
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : healthScore !== null && healthScore >= 0.5
                    ? 'text-amber-600 dark:text-amber-400'
                    : 'text-rose-600 dark:text-rose-400')}
              {statCard('Active improvements', activeImprovements, 'text-blue-600 dark:text-blue-400')}
              {statCard('Open incidents', openIncidents,
                openIncidents > 0 ? 'text-rose-600 dark:text-rose-400' : 'text-slate-900 dark:text-white')}
              {statCard('Open alerts', openAlerts,
                openAlerts > 0 ? 'text-amber-600 dark:text-amber-400' : 'text-slate-900 dark:text-white')}
            </div>

            {/* System health + dependency graph */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {panel('System health', health ? (
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <p className="text-4xl font-bold text-slate-900 dark:text-white">
                      {healthScore !== null ? Math.round(healthScore * 100) : '—'}
                    </p>
                    {badge(String(health.status ?? ''))}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    {Object.entries((health.signals as Dict) ?? {}).map(([k, v]) => (
                      <div key={k} className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                        <span className="text-slate-500 dark:text-slate-400">{k.replace(/_/g, ' ')}</span>
                        <span className="font-medium text-slate-800 dark:text-slate-200">{Math.round(Number(v) * 100)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : emptyRow('System health unavailable'))}
              {panel('Service dependencies', deps ? (
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">
                    {((deps.services as string[]) ?? []).length} services in the dependency graph
                  </p>
                  <ul className="text-sm space-y-1">
                    {((deps.edges as Array<{ from: string; to: string }>) ?? []).slice(0, 24).map((e, i) => (
                      <li key={i} className="rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-1.5">
                        <span className="font-medium text-slate-800 dark:text-slate-200">{e.from}</span>
                        <span className="text-slate-400 mx-1">→</span>
                        <span className="text-slate-600 dark:text-slate-300">{e.to}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : emptyRow('Dependency graph unavailable'))}
            </div>

            {/* Incidents + alerts */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {panel('Incidents', incidents.length ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <th className="pb-2 pr-3">Severity</th>
                      <th className="pb-2 pr-3">Title</th>
                      <th className="pb-2">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {incidents.slice(0, 8).map((i) => (
                      <tr key={String(i.id)} className="border-t border-slate-100 dark:border-slate-700">
                        <td className="py-2 pr-3">{badge(String(i.severity ?? ''))}</td>
                        <td className="py-2 pr-3 text-slate-800 dark:text-slate-200">{String(i.title ?? '')}</td>
                        <td className="py-2">{badge(String(i.status ?? ''))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : emptyRow('No incidents recorded'))}
              {panel('Alerts', alerts.length ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <th className="pb-2 pr-3">Severity</th>
                      <th className="pb-2 pr-3">Category</th>
                      <th className="pb-2 pr-3">Message</th>
                      <th className="pb-2">Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {alerts.slice(0, 8).map((a) => (
                      <tr key={String(a.id)} className="border-t border-slate-100 dark:border-slate-700">
                        <td className="py-2 pr-3">{badge(String(a.severity ?? ''))}</td>
                        <td className="py-2 pr-3 text-slate-600 dark:text-slate-300">{String(a.category ?? '')}</td>
                        <td className="py-2 pr-3 text-slate-800 dark:text-slate-200">{String(a.message ?? '')}</td>
                        <td className="py-2 text-slate-600 dark:text-slate-300">{String(a.occurrence_count ?? 1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : emptyRow('No alerts'))}
            </div>

            {/* Improvements + experiments */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {panel('Improvement proposals', improvements.length ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <th className="pb-2 pr-3">Title</th>
                      <th className="pb-2 pr-3">Domain</th>
                      <th className="pb-2">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {improvements.slice(0, 8).map((p) => (
                      <tr key={String(p.id)} className="border-t border-slate-100 dark:border-slate-700">
                        <td className="py-2 pr-3 text-slate-800 dark:text-slate-200">{String(p.title ?? '')}</td>
                        <td className="py-2 pr-3 text-slate-600 dark:text-slate-300">{String(p.domain ?? '')}</td>
                        <td className="py-2">{badge(String(p.status ?? ''))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : emptyRow('No improvement proposals'))}
              {panel('Experiments', experiments.length ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <th className="pb-2 pr-3">Name</th>
                      <th className="pb-2 pr-3">Domain</th>
                      <th className="pb-2">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {experiments.slice(0, 8).map((e) => (
                      <tr key={String(e.id)} className="border-t border-slate-100 dark:border-slate-700">
                        <td className="py-2 pr-3 text-slate-800 dark:text-slate-200">{String(e.name ?? '')}</td>
                        <td className="py-2 pr-3 text-slate-600 dark:text-slate-300">{String(e.domain ?? '')}</td>
                        <td className="py-2">{badge(String(e.status ?? ''))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : emptyRow('No experiments'))}
            </div>

            {/* SLO + governance */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {panel('SLO history', slo.length ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <th className="pb-2 pr-3">SLO</th>
                      <th className="pb-2 pr-3">Measured</th>
                      <th className="pb-2 pr-3">Target</th>
                      <th className="pb-2">Met</th>
                    </tr>
                  </thead>
                  <tbody>
                    {slo.slice(0, 8).map((s) => (
                      <tr key={String(s.id)} className="border-t border-slate-100 dark:border-slate-700">
                        <td className="py-2 pr-3 text-slate-800 dark:text-slate-200">{String(s.name ?? '')}</td>
                        <td className="py-2 pr-3 text-slate-600 dark:text-slate-300">{String(s.measured_value ?? '—')}</td>
                        <td className="py-2 pr-3 text-slate-600 dark:text-slate-300">{String(s.target ?? '—')}</td>
                        <td className="py-2">{s.met === true ? badge('RESOLVED') : s.met === false ? badge('OPEN') : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : emptyRow('No SLO history'))}
              {panel('Governance changes', govAudit.length ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <th className="pb-2 pr-3">Version</th>
                      <th className="pb-2 pr-3">Type</th>
                      <th className="pb-2 pr-3">Reason</th>
                      <th className="pb-2">Changed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {govAudit.slice(0, 8).map((g) => (
                      <tr key={String(g.version)} className="border-t border-slate-100 dark:border-slate-700">
                        <td className="py-2 pr-3 text-slate-600 dark:text-slate-300">v{String(g.version ?? '')}</td>
                        <td className="py-2 pr-3 text-slate-600 dark:text-slate-300">{String(g.policy_type ?? '')}</td>
                        <td className="py-2 pr-3 text-slate-800 dark:text-slate-200">{String(g.reason ?? '')}</td>
                        <td className="py-2 text-slate-600 dark:text-slate-300">{String((g.changed_keys as string[])?.length ?? 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : emptyRow('No governance changes'))}
            </div>

            {/* Knowledge health + quality + safety */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {panel('Knowledge health', knowledge ? (
                <div className="text-sm space-y-2">
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Health score</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">{String(knowledge.score ?? '—')}</span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Scope</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">{String(knowledge.scope_type ?? '—')}</span>
                  </div>
                  {gaps && Array.isArray(gaps.gaps) && (gaps.gaps as Dict[]).length > 0 && (
                    <div className="pt-1">
                      <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-1">Knowledge gaps</p>
                      {(gaps.gaps as Dict[]).slice(0, 4).map((g) => (
                        <p key={String(g.id)} className="text-slate-700 dark:text-slate-300">
                          • {String(g.query ?? '')}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              ) : emptyRow('Knowledge health unavailable'))}
              {panel('AI quality scorecards', scorecards.length ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <th className="pb-2 pr-3">Domain</th>
                      <th className="pb-2">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scorecards.slice(0, 6).map((c) => (
                      <tr key={String(c.domain)} className="border-t border-slate-100 dark:border-slate-700">
                        <td className="py-2 pr-3 text-slate-800 dark:text-slate-200">{String(c.domain ?? '')}</td>
                        <td className="py-2 text-slate-600 dark:text-slate-300">{String(c.score ?? c.overall_score ?? '—')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : emptyRow('No scorecards'))}
              {panel('AI safety', safety ? (
                <div className="text-sm space-y-2">
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Injection detection</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {detectionRate !== null ? `${detectionRate}%` : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Corpus cases</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">{String(safety.cases ?? '—')}</span>
                  </div>
                  <p className="text-xs text-slate-500 dark:text-slate-400 pt-1">
                    Prompt-injection regression corpus — governance stays human-owned.
                  </p>
                </div>
              ) : emptyRow('Safety scan unavailable'))}
            </div>

            {/* Intelligence + feedback + API/DB health */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {panel('Agent & workflow intelligence', (
                <div className="text-sm space-y-2">
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Agent success rate</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {agents ? `${Math.round(Number(((agents.success as Dict)?.success_rate ?? 0)) * 100)}%` : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Agent safety score</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {agents ? String(((agents.safety as Dict)?.safety_score ?? '—')) : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Workflow completion</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {workflows ? `${Math.round(Number(((workflows.analytics as Dict)?.completion_rate ?? 0)) * 100)}%` : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Workflow bottlenecks</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {workflows ? String((workflows.bottlenecks as Dict[])?.length ?? 0) : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Memory health</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {memory ? `${Math.round(Number(((memory.quality as Dict)?.health_fraction ?? 0)) * 100)}%` : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Graph health</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {graph ? String(((graph.health as Dict)?.score ?? '—')) : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Search quality</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {search && typeof search.score === 'number' ? String(search.score) : '—'}
                    </span>
                  </div>
                </div>
              ))}
              {panel('Feedback', feedback ? (
                <div className="text-sm space-y-2">
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Total</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">{String(feedback.total ?? 0)}</span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Positive</span>
                    <span className="font-medium text-emerald-600 dark:text-emerald-400">{String(feedback.positive ?? 0)}</span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Negative</span>
                    <span className="font-medium text-rose-600 dark:text-rose-400">{String(feedback.negative ?? 0)}</span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Flagged</span>
                    <span className="font-medium text-amber-600 dark:text-amber-400">{String(feedback.flagged ?? 0)}</span>
                  </div>
                </div>
              ) : emptyRow('No feedback recorded'))}
              {panel('API & database health', (
                <div className="text-sm space-y-2">
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">API error rate</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {apiHealth ? `${Math.round(Number(apiHealth.error_rate ?? 0) * 100)}%` : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Unstable endpoints</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {apiHealth ? String((apiHealth.unstable as Dict[])?.length ?? 0) : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Migration head</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {dbHealth ? String(((dbHealth.migration as Dict)?.migration_head ?? '—')) : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Tracked DB rows</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {dbHealth ? String(((dbHealth.growth as Dict)?.total_rows ?? 0)) : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-slate-50 dark:bg-slate-900 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Versioned reports</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">{reports.length}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {panel('Phase 23 — global AI cloud', p23Providers === null && p23Vector === null
          ? emptyRow('Phase 23 signals unavailable')
          : (
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-lg bg-slate-50 dark:bg-slate-900 p-4">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-3">Provider readiness</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between rounded-lg bg-white dark:bg-slate-800 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Environment</span>
                    <span className="font-mono font-medium text-slate-800 dark:text-slate-100">{String((p23Providers?.environment as string) ?? '—')}</span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-white dark:bg-slate-800 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Providers validated</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">{Array.isArray(p23Providers?.providers) ? (p23Providers.providers as unknown[]).length : 0}</span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-white dark:bg-slate-800 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Synthetic validation</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {p23Providers?.real_provider_available ? 'REAL' : 'SIMULATED (fake provider)'}
                    </span>
                  </div>
                </div>
              </div>
              <div className="rounded-lg bg-slate-50 dark:bg-slate-900 p-4">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-3">Vector platform</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between rounded-lg bg-white dark:bg-slate-800 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Backend</span>
                    <span className="font-mono font-medium text-slate-800 dark:text-slate-100">{String((p23Vector?.backend as string) ?? '—')}</span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-white dark:bg-slate-800 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Realization</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">{String((p23Vector?.realization as string) ?? '—')}</span>
                  </div>
                  <div className="flex justify-between rounded-lg bg-white dark:bg-slate-800 px-3 py-2">
                    <span className="text-slate-500 dark:text-slate-400">Coverage</span>
                    <span className="font-medium text-slate-800 dark:text-slate-200">{p23Vector && p23Vector.coverage_ratio !== undefined ? `${Math.round(Number(p23Vector.coverage_ratio) * 100)}%` : '—'}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
      </div>
    </main>
  );
}