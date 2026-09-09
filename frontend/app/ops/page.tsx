'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import api from '@/lib/api';

interface QueueMetrics {
  by_queue: Array<{
    queue_name: string;
    depth: number;
    running: number;
    waiting: number;
    retrying: number;
    dead: number;
  }>;
  by_tenant: Array<{ workspace_id: number | null; active: number; queued: number }>;
}

interface VectorStatus {
  backend: string;
  pgvector_available: boolean;
  json_fallback: boolean;
  note?: string;
}

interface ProviderHealth {
  items: Array<{
    id: number;
    provider: string;
    model: string;
    status: string;
    circuit_state: string;
    consecutive_failures: number;
    success_count: number;
    failure_count: number;
    avg_latency_ms: number | null;
    last_error: string | null;
    last_checked_at: string | null;
  }>;
}

interface GatewayStatus {
  status?: string;
  message?: string;
  primary_provider?: string;
  fallback_configured?: boolean;
  [key: string]: unknown;
}

interface FleetWorker {
  worker_id: string;
  status: string;
  hostname: string | null;
  pid: number | null;
  version: string | null;
  queue_name: string | null;
  current_job_type: string | null;
  current_job_id: number | null;
  started_at: string | null;
  last_heartbeat: string | null;
}

interface DeadLetterItem {
  id: number;
  queue_name: string;
  job_type: string;
  workspace_id: number;
  attempt: number;
  max_attempts: number;
  error_message: string | null;
}

function statCard(label: string, value: string | number, accent: string) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 shadow-sm">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
      <p className={`mt-1 text-2xl font-bold ${accent}`}>{value}</p>
    </div>
  );
}

function badge(status: string) {
  const map: Record<string, string> = {
    healthy: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
    degraded: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
    down: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
    open: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
    half_open: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
    closed: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
    running: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
    idle: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
    starting: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
    stopping: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
    stopped: 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300',
    dead: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
    up: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
    ok: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
    breached: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
    no_data: 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300',
    quarantined: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
    unknown: 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300',
  };
  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-medium rounded-full ${map[status] ?? map.unknown}`}>
      {status.replace('_', ' ')}
    </span>
  );
}

export default function OperationsPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  const [workspaceId, setWorkspaceId] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<QueueMetrics | null>(null);
  const [vector, setVector] = useState<VectorStatus | null>(null);
  const [providers, setProviders] = useState<ProviderHealth | null>(null);
  const [gateway, setGateway] = useState<GatewayStatus | null>(null);
  const [fleet, setFleet] = useState<FleetWorker[] | null>(null);
  const [signals, setSignals] = useState<Record<string, unknown> | null>(null);
  const [deadLetters, setDeadLetters] = useState<DeadLetterItem[] | null>(null);
  const [backfillPreview, setBackfillPreview] = useState<Record<string, unknown> | null>(null);
  const [workerHealth, setWorkerHealth] = useState<Record<string, unknown> | null>(null);
  const [brokerHealth, setBrokerHealth] = useState<Record<string, unknown> | null>(null);
  const [vectorHealth, setVectorHealth] = useState<Record<string, unknown> | null>(null);
  const [poisonDocs, setPoisonDocs] = useState<Array<Record<string, unknown>> | null>(null);
  const [connectors, setConnectors] = useState<Array<Record<string, unknown>> | null>(null);
  const [slo, setSlo] = useState<Record<string, unknown> | null>(null);
  const [costAttr, setCostAttr] = useState<Record<string, unknown> | null>(null);
  const [costProj, setCostProj] = useState<Record<string, unknown> | null>(null);
  const [searchStats, setSearchStats] = useState<Record<string, unknown> | null>(null);
  const [backupInv, setBackupInv] = useState<Record<string, unknown> | null>(null);
  const [restoreCheck, setRestoreCheck] = useState<Record<string, unknown> | null>(null);
  const [globalHealth, setGlobalHealth] = useState<Record<string, unknown> | null>(null);
  const [controlConfig, setControlConfig] = useState<Record<string, unknown> | null>(null);
  const [regions, setRegions] = useState<Array<Record<string, unknown>> | null>(null);
  const [quarantines, setQuarantines] = useState<Array<Record<string, unknown>> | null>(null);
  const [broker2, setBroker2] = useState<Record<string, unknown> | null>(null);
  const [schedulerLeader, setSchedulerLeader] = useState<Record<string, unknown> | null>(null);
  const [capabilities, setCapabilities] = useState<Record<string, unknown> | null>(null);
  const [vectorCoverage, setVectorCoverage] = useState<Record<string, unknown> | null>(null);
  const [vectorDrift, setVectorDrift] = useState<Record<string, unknown> | null>(null);
  const [ingestionGovernor, setIngestionGovernor] = useState<Record<string, unknown> | null>(null);
  const [slo19, setSlo19] = useState<Record<string, unknown> | null>(null);
  const [consistency, setConsistency] = useState<Record<string, unknown> | null>(null);
  const [drReadiness, setDrReadiness] = useState<Record<string, unknown> | null>(null);
  const [searchQuality, setSearchQuality] = useState<Record<string, unknown> | null>(null);
  const [evaluations, setEvaluations] = useState<Array<Record<string, unknown>> | null>(null);
  const [p22infra, setP22infra] = useState<Record<string, unknown> | null>(null);
  const [p22backend, setP22backend] = useState<Record<string, unknown> | null>(null);
  const [p22drift, setP22drift] = useState<Record<string, unknown> | null>(null);
  const [p22regions, setP22regions] = useState<Record<string, unknown> | null>(null);
  const [p22dr, setP22dr] = useState<Record<string, unknown> | null>(null);
  const [p22streams, setP22streams] = useState<Record<string, unknown> | null>(null);
  const [p23health, setP23health] = useState<Record<string, unknown> | null>(null);
  const [p23capabilities, setP23capabilities] = useState<Record<string, unknown>[] | null>(null);
  const [p23providers, setP23providers] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadAll = useCallback(async () => {
    if (!workspaceId) return;
    setRefreshing(true);
    setError(null);
    setNotice(null);
    const collected: string[] = [];
    try {
      const [vectorData, queueData, providerData, gatewayData, fleetData,
        signalData, deadData, previewData, whData, bhData, vhData, poisonData,
        connData, sloData, attrData, projData, searchData, invData, restoreData] = await Promise.all([
        api.getVectorStatus(),
        api.getQueueMetrics(workspaceId),
        api.getProviderStatus(workspaceId),
        api.getGatewayStatus(workspaceId),
        api.getWorkerFleet(),
        api.getAutoscaleSignals(),
        api.listDeadLetters(workspaceId),
        api.getVectorBackfillPreview(),
        api.getWorkerHealth(),
        api.getBrokerHealth(),
        api.getVectorHealth(workspaceId),
        api.getPoisonDocuments(workspaceId),
        api.getConnectorHealth(workspaceId),
        api.getSloStatus(workspaceId),
        api.getCostAttribution(workspaceId),
        api.getCostProjection(workspaceId),
        api.getSearchAnalytics(workspaceId),
        api.getBackupInventory(workspaceId),
        api.validateRestore(workspaceId),
      ]);
      setVector(vectorData);
      setMetrics(queueData);
      setProviders(providerData);
      setGateway(gatewayData);
      setFleet(fleetData.items);
      setSignals(signalData);
      setDeadLetters(deadData.items);
      setBackfillPreview(previewData);
      setWorkerHealth(whData);
      setBrokerHealth(bhData);
      setVectorHealth(vhData);
      setPoisonDocs(poisonData.items);
      setConnectors(connData.items);
      setSlo(sloData);
      setCostAttr(attrData);
      setCostProj(projData);
      setSearchStats(searchData);
      setBackupInv(invData);
      setRestoreCheck(restoreData);
      // Phase 19 control-plane panels — fetch softly so one failure never blanks the page
      const p19 = await Promise.allSettled([
        api.getOps19Health(),
        api.getControlConfig(),
        api.getRegions(),
        api.getQuarantines(),
        api.getBroker2Info(),
        api.getSchedulerLeader(),
        api.getProviderCapabilities(),
        api.getVectorCoverage(workspaceId),
        api.getVectorDrift(),
        api.getIngestionGovernor(),
        api.getSloHealth(),
        api.runConsistencyCheck(workspaceId, true),
        api.getDrReadiness(),
        api.getSearchQuality(workspaceId),
        api.getQualityEvaluations(),
      ]);
      const [gh, cc, rg, qr, bk, sl, cap, vc, vd, ig, slo, cs, dr, sq, ev] = p19;
      if (gh.status === 'fulfilled') setGlobalHealth(gh.value);
      if (cc.status === 'fulfilled') setControlConfig(cc.value);
      if (rg.status === 'fulfilled') setRegions(rg.value);
      if (qr.status === 'fulfilled') setQuarantines(qr.value);
      if (bk.status === 'fulfilled') setBroker2(bk.value);
      if (sl.status === 'fulfilled') setSchedulerLeader(sl.value);
      if (cap.status === 'fulfilled') setCapabilities(cap.value);
      if (vc.status === 'fulfilled') setVectorCoverage(vc.value);
      if (vd.status === 'fulfilled') setVectorDrift(vd.value);
      if (ig.status === 'fulfilled') setIngestionGovernor(ig.value);
      if (slo.status === 'fulfilled') setSlo19(slo.value);
      if (cs.status === 'fulfilled') setConsistency(cs.value);
      if (dr.status === 'fulfilled') setDrReadiness(dr.value);
      if (sq.status === 'fulfilled') setSearchQuality(sq.value);
      if (ev.status === 'fulfilled') setEvaluations(ev.value);
      if (!vectorData.pgvector_available) {
        collected.push('JSON vector fallback active — pgvector extension not installed');
      }
      // Phase 22 production-cloud panels — soft fetch so one failure never blanks the page
      const p22 = await Promise.allSettled([
        api.getInfrastructure(),
        api.getVectorBackend(),
        api.getVectorDriftP22(workspaceId),
        api.getRegionsP22(workspaceId),
        api.getDrReport(workspaceId),
        api.getOpsStream(workspaceId, 'ops'),
      ]);
      const [p22inf, p22vec, p22dr, p22reg, p22drr, p22str] = p22;
      if (p22inf.status === 'fulfilled') setP22infra(p22inf.value);
      if (p22vec.status === 'fulfilled') setP22backend(p22vec.value);
      if (p22dr.status === 'fulfilled') setP22drift(p22dr.value);
      if (p22reg.status === 'fulfilled') setP22regions(p22reg.value);
      if (p22drr.status === 'fulfilled') setP22dr(p22drr.value);
      if (p22str.status === 'fulfilled') setP22streams(p22str.value);
      if (p22vec.status === 'fulfilled') {
        const vb = p22vec.value as Record<string, unknown>;
        if (vb.backend === 'json_fallback') collected.push('Vector benchmark runs in simulated mode (JSON fallback)');
      }
      // Phase 23 global-cloud panels — soft fetch, one failure never blanks the page
      const p23 = await Promise.allSettled([
        api.getGlobalHealth(),
        api.getCapabilities(),
        api.getProviderReadiness(),
      ]);
      if (p23[0].status === 'fulfilled') setP23health(p23[0].value);
      if (p23[1].status === 'fulfilled') setP23capabilities(p23[1].value);
      if (p23[2].status === 'fulfilled') setP23providers(p23[2].value);
      const gw = gatewayData as GatewayStatus;
      if (gw.message) collected.push(String(gw.message));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load operations view');
    } finally {
      if (collected.length) setNotice(collected.join('. '));
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
        else setError('No workspace available for operations view');
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

  const totalDepth = metrics?.by_queue.reduce((n, q) => n + q.depth, 0) ?? 0;
  const totalDead = metrics?.by_queue.reduce((n, q) => n + q.dead, 0) ?? 0;
  const totalRetrying = metrics?.by_queue.reduce((n, q) => n + q.retrying, 0) ?? 0;
  const totalRunning = metrics?.by_queue.reduce((n, q) => n + q.running, 0) ?? 0;

  return (
    <main className="min-h-screen bg-slate-50 dark:bg-slate-900">
      <div className="max-w-6xl mx-auto px-4 py-8">
        <header className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Operations</h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Worker fleet, broker, providers, vector platform, ingestion, connectors, SLOs, cost, search, and disaster recovery
            </p>
          </div>
          <div className="flex gap-2">
            <a
              href="/autonomy"
              className="px-3 py-1.5 rounded-lg text-sm font-medium text-emerald-700 dark:text-emerald-300 bg-white dark:bg-slate-800 border border-emerald-300 dark:border-emerald-800 hover:border-emerald-500"
            >
              Autonomy Center
            </a>
            <a
              href="/ai-control"
              className="px-3 py-1.5 rounded-lg text-sm font-medium text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-blue-400"
            >
              AI Control Center
            </a>
            <a
              href="/intelligence"
              className="px-3 py-1.5 rounded-lg text-sm font-medium text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-blue-400"
            >
              Knowledge OS
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
            {error} {/admin required|owner required|forbidden|403/i.test(error) ? '(operations endpoints require an owner or organization-admin role)' : ''}
          </div>
        )}
        {notice && (
          <div className="mb-4 px-4 py-3 rounded-lg bg-blue-50 dark:bg-blue-950/50 border border-blue-200 dark:border-blue-800 text-sm text-blue-800 dark:text-blue-200">
            {notice}
          </div>
        )}

        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
          {statCard('Queued jobs', totalDepth, 'text-blue-600 dark:text-blue-400')}
          {statCard('Running', totalRunning, 'text-indigo-600 dark:text-indigo-400')}
          {statCard('Retrying', totalRetrying, 'text-amber-600 dark:text-amber-400')}
          {statCard('Dead letter', totalDead, totalDead > 0 ? 'text-red-600 dark:text-red-400' : 'text-slate-600 dark:text-slate-400')}
          {statCard(
            'Vector backend',
            vector?.pgvector_available ? 'pgvector' : 'JSON fallback',
            vector?.pgvector_available
              ? 'text-emerald-600 dark:text-emerald-400'
              : 'text-amber-600 dark:text-amber-400'
          )}
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Queue depth by class</h2>
            {metrics && metrics.by_queue.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                    <th className="pb-2">Queue</th>
                    <th className="pb-2">Waiting</th>
                    <th className="pb-2">Running</th>
                    <th className="pb-2">Retrying</th>
                    <th className="pb-2">Dead</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.by_queue.map((q) => (
                    <tr key={q.queue_name} className="border-t border-slate-100 dark:border-slate-700">
                      <td className="py-2 font-medium text-slate-700 dark:text-slate-200">{q.queue_name}</td>
                      <td className="py-2 text-slate-600 dark:text-slate-300">{q.waiting}</td>
                      <td className="py-2 text-slate-600 dark:text-slate-300">{q.running}</td>
                      <td className="py-2 text-slate-600 dark:text-slate-300">{q.retrying}</td>
                      <td className="py-2 text-slate-600 dark:text-slate-300">{q.dead}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">No queued work.</p>
            )}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Tenant load (active jobs)</h2>
            {metrics && metrics.by_tenant.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                    <th className="pb-2">Workspace</th>
                    <th className="pb-2">Queued</th>
                    <th className="pb-2">Active</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.by_tenant.slice(0, 10).map((t) => (
                    <tr key={String(t.workspace_id)} className="border-t border-slate-100 dark:border-slate-700">
                      <td className="py-2 font-medium text-slate-700 dark:text-slate-200">
                        #{t.workspace_id ?? 'system'}
                      </td>
                      <td className="py-2 text-slate-600 dark:text-slate-300">{t.queued}</td>
                      <td className="py-2 text-slate-600 dark:text-slate-300">{t.active}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">No tenant load data.</p>
            )}
          </section>
        </div>

        <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm mt-6">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Provider health</h2>
          {providers && providers.items.length > 0 ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  <th className="pb-2">Provider</th>
                  <th className="pb-2">Model</th>
                  <th className="pb-2">Status</th>
                  <th className="pb-2">Circuit</th>
                  <th className="pb-2">Success</th>
                  <th className="pb-2">Failures</th>
                  <th className="pb-2">Avg latency</th>
                </tr>
              </thead>
              <tbody>
                {providers.items.map((p) => (
                  <tr key={p.id} className="border-t border-slate-100 dark:border-slate-700">
                    <td className="py-2 font-medium text-slate-700 dark:text-slate-200">{p.provider}</td>
                    <td className="py-2 text-slate-600 dark:text-slate-300">{p.model}</td>
                    <td className="py-2">{badge(p.status)}</td>
                    <td className="py-2">{badge(p.circuit_state)}</td>
                    <td className="py-2 text-slate-600 dark:text-slate-300">{p.success_count}</td>
                    <td className="py-2 text-slate-600 dark:text-slate-300">{p.failure_count}</td>
                    <td className="py-2 text-slate-600 dark:text-slate-300">
                      {p.avg_latency_ms != null ? `${p.avg_latency_ms} ms` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-sm text-slate-500 dark:text-slate-400">No provider health records yet — no real provider configured; deterministic fake provider is in use.</p>
          )}
        </section>

        {gateway && (gateway.status || gateway.primary_provider) && (
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm mt-6">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">AI gateway</h2>
            <dl className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Status</dt>
                <dd className="mt-1 text-slate-800 dark:text-slate-100">{String(gateway.status ?? '—')}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Primary provider</dt>
                <dd className="mt-1 text-slate-800 dark:text-slate-100">{String(gateway.primary_provider ?? '—')}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Fallback configured</dt>
                <dd className="mt-1 text-slate-800 dark:text-slate-100">
                  {gateway.fallback_configured ? 'Yes' : 'No'}
                </dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Message</dt>
                <dd className="mt-1 text-slate-800 dark:text-slate-100">{gateway.message ? String(gateway.message) : '—'}</dd>
              </div>
            </dl>
          </section>
        )}

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Worker fleet</h2>
            {fleet && fleet.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                    <th className="pb-2">Worker</th>
                    <th className="pb-2">Status</th>
                    <th className="pb-2">Queue</th>
                    <th className="pb-2">Job</th>
                    <th className="pb-2">Last beat</th>
                  </tr>
                </thead>
                <tbody>
                  {fleet.map((w) => (
                    <tr key={w.worker_id} className="border-t border-slate-100 dark:border-slate-700">
                      <td className="py-2 font-medium text-slate-700 dark:text-slate-200 font-mono text-xs">{w.worker_id}</td>
                      <td className="py-2">{badge(w.status.toLowerCase())}</td>
                      <td className="py-2 text-slate-600 dark:text-slate-300">{w.queue_name ?? '—'}</td>
                      <td className="py-2 text-slate-600 dark:text-slate-300">
                        {w.current_job_type ? `${w.current_job_type}${w.current_job_id ? ` #${w.current_job_id}` : ''}` : 'idle'}
                      </td>
                      <td className="py-2 text-slate-500 dark:text-slate-400 text-xs">
                        {w.last_heartbeat ? new Date(w.last_heartbeat).toLocaleTimeString() : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">No active workers — start a worker process to drain the queue.</p>
            )}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Autoscale signals</h2>
            {signals ? (
              <dl className="grid grid-cols-2 gap-4 text-sm">
                {['queue_depth', 'running', 'retrying', 'dead_lettered',
                  'worker_count', 'failure_rate'].map((key) => (
                  <div key={key}>
                    <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">{key.replace(/_/g, ' ')}</dt>
                    <dd className="mt-1 text-slate-800 dark:text-slate-100 font-semibold">
                      {signals[key] != null ? String(signals[key]) : '—'}
                    </dd>
                  </div>
                ))}
                <div className="col-span-2">
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Oldest job age</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100 font-semibold">
                    {signals.oldest_job_age_seconds != null
                      ? `${Number(signals.oldest_job_age_seconds)}s`
                      : 'no queued work'}
                  </dd>
                </div>
              </dl>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">Signals unavailable.</p>
            )}
          </section>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Dead letters</h2>
            {deadLetters && deadLetters.length > 0 ? (
              <ul className="divide-y divide-slate-100 dark:divide-slate-700 text-sm">
                {deadLetters.slice(0, 6).map((j) => (
                  <li key={j.id} className="py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-slate-700 dark:text-slate-200 font-mono text-xs">
                        {j.queue_name}/{j.job_type} #{j.id}
                      </span>
                      <span className="text-xs text-red-600 dark:text-red-400">attempt {j.attempt}/{j.max_attempts}</span>
                    </div>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 truncate">{(j.error_message ?? '').slice(0, 160)}</p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">No dead-lettered jobs.</p>
            )}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Vector backfill</h2>
            {backfillPreview ? (
              <div className="text-sm space-y-2">
                <div className="flex justify-between"><span className="text-slate-500 dark:text-slate-400">Backend</span><span className="font-medium text-slate-800 dark:text-slate-100">{String(backfillPreview.vector_backend)}</span></div>
                <div className="flex justify-between"><span className="text-slate-500 dark:text-slate-400">Model</span><span className="font-medium text-slate-800 dark:text-slate-100 font-mono text-xs">{String(backfillPreview.model)}</span></div>
                <div className="flex justify-between"><span className="text-slate-500 dark:text-slate-400">Dimensions</span><span className="font-medium text-slate-800 dark:text-slate-100">{String(backfillPreview.dimensions)}</span></div>
                <div className="flex justify-between"><span className="text-slate-500 dark:text-slate-400">Chunks to embed</span><span className="font-medium text-slate-800 dark:text-slate-100">{String(backfillPreview.chunks_to_embed)}</span></div>
                <div className="flex justify-between"><span className="text-slate-500 dark:text-slate-400">Documents</span><span className="font-medium text-slate-800 dark:text-slate-100">{String(backfillPreview.documents_to_process)}</span></div>
                <p className="pt-1 text-xs text-slate-500 dark:text-slate-400">
                  Backfill is an operator-triggered, batch, idempotent process — never runs automatically.
                </p>
              </div>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">Preview unavailable.</p>
            )}
          </section>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Broker health</h2>
            {brokerHealth ? (
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Broker</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100 font-semibold">{String(brokerHealth.broker)}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Status</dt>
                  <dd className="mt-1">{badge(String(brokerHealth.status).toLowerCase())}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Latency</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">
                    {brokerHealth.latency_ms != null ? `${Number(brokerHealth.latency_ms)} ms` : '—'}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Queue depth</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">{String(brokerHealth.queue_depth)}</dd>
                </div>
              </dl>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">Broker health unavailable.</p>
            )}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Vector platform health</h2>
            {vectorHealth ? (
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Backend</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100 font-semibold">{String(vectorHealth.backend)}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Model</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100 font-mono text-xs">
                    {String((vectorHealth.active_model as Record<string, unknown> | null)?.model ?? 'none')}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Dimensions</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">
                    {String((vectorHealth.active_model as Record<string, unknown> | null)?.dimensions ?? '—')}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Vector coverage</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">
                    {String(vectorHealth.vector_coverage_pct)}% ({String(vectorHealth.total_chunks)} chunks)
                  </dd>
                </div>
              </dl>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">Vector health unavailable.</p>
            )}
          </section>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Ingestion poison documents</h2>
            {poisonDocs && poisonDocs.length > 0 ? (
              <ul className="divide-y divide-slate-100 dark:divide-slate-700 text-sm">
                {poisonDocs.slice(0, 5).map((p) => (
                  <li key={String(p.id)} className="py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-slate-700 dark:text-slate-200 font-mono text-xs">
                        doc #{String(p.document_id ?? '—')} · {String(p.stage)}
                      </span>
                      <span className="text-xs text-red-600 dark:text-red-400">{String(p.failure_count)} failures</span>
                    </div>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 truncate">{String(p.last_error ?? '').slice(0, 140)}</p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">No quarantined documents — ingestion is healthy.</p>
            )}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Knowledge connectors</h2>
            {connectors && connectors.length > 0 ? (
              <ul className="divide-y divide-slate-100 dark:divide-slate-700 text-sm">
                {connectors.slice(0, 5).map((c) => (
                  <li key={String(c.id)} className="py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-slate-700 dark:text-slate-200">{String(c.name)}</span>
                      <span className="text-xs">{badge(String(c.enabled ? 'running' : 'stopped'))}</span>
                    </div>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      {String(c.kind)} · last sync: {String(c.last_sync_status ?? 'never')}
                      {c.lag_s != null ? ` · lag ${Number(c.lag_s)}s` : ''} · {String(c.total_items)} items
                    </p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">No connectors configured — external sources are optional.</p>
            )}
          </section>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">SLO compliance</h2>
            {slo ? (
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Status</dt>
                  <dd className="mt-1">{badge(String(slo.status).toLowerCase())}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Snapshots</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">{String(slo.snapshots)}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Availability</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">
                    {slo.availability != null ? `${(Number(slo.availability) * 100).toFixed(2)}%` : '—'}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Error rate</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">
                    {slo.error_rate != null ? `${(Number(slo.error_rate) * 100).toFixed(2)}%` : '—'}
                  </dd>
                </div>
              </dl>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">SLO data unavailable.</p>
            )}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Search analytics</h2>
            {searchStats ? (
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Searches</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">{String(searchStats.searches)}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Zero-result rate</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">
                    {searchStats.zero_result_rate != null ? `${(Number(searchStats.zero_result_rate) * 100).toFixed(1)}%` : '—'}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">p95 latency</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">
                    {searchStats.p95_latency_ms != null ? `${Number(searchStats.p95_latency_ms)} ms` : '—'}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Avg latency</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">
                    {searchStats.avg_latency_ms != null ? `${Number(searchStats.avg_latency_ms)} ms` : '—'}
                  </dd>
                </div>
              </dl>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">No search analytics yet.</p>
            )}
          </section>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Cost attribution (30d)</h2>
            {costAttr ? (
              <div className="text-sm space-y-3">
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Total estimated cost</span>
                  <span className="font-semibold text-slate-800 dark:text-slate-100">${Number(costAttr.total_cost).toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Executions</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">{String(costAttr.execution_count)}</span>
                </div>
                {(costAttr.by_feature as Array<{ key: string; cost: number }> | undefined)?.slice(0, 4).map((f) => (
                  <div key={String(f.key)} className="flex justify-between">
                    <span className="text-slate-500 dark:text-slate-400 font-mono text-xs">{String(f.key)}</span>
                    <span className="font-medium text-slate-800 dark:text-slate-100">${f.cost.toFixed(4)}</span>
                  </div>
                ))}
                <p className="pt-1 text-xs text-slate-500 dark:text-slate-400">
                  Estimated from provider usage accounting; projections below are labeled estimates.
                </p>
              </div>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">Cost data unavailable.</p>
            )}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Cost projection (estimate)</h2>
            {costProj ? (
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Last 30d cost</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">${Number(costProj.last_30d_cost).toFixed(4)}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Daily rate (est)</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">${Number(costProj.daily_rate_est).toFixed(4)}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">30d projection (est)</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">${Number(costProj.projected_30d_cost_est).toFixed(4)}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Confidence</dt>
                  <dd className="mt-1 text-slate-800 dark:text-slate-100">{String(costProj.confidence)}</dd>
                </div>
              </dl>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">Projection unavailable.</p>
            )}
          </section>
        </div>

        <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm mt-6">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Disaster recovery</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="text-sm space-y-2">
              {backupInv ? (
                <>
                  <div className="flex justify-between">
                    <span className="text-slate-500 dark:text-slate-400">Database backup</span>
                    <span className="font-medium text-slate-800 dark:text-slate-100">
                      {String((backupInv.database as Record<string, unknown>)?.backup_method ?? '—')} · {String((backupInv.database as Record<string, unknown>)?.schedule ?? '—')}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500 dark:text-slate-400">Object storage</span>
                    <span className="font-medium text-slate-800 dark:text-slate-100">
                      {(backupInv.object_storage as Record<string, unknown>)?.configured ? 'configured' : 'not configured'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500 dark:text-slate-400">Migrations</span>
                    <span className="font-medium text-slate-800 dark:text-slate-100 font-mono text-xs">
                      {String((backupInv.migrations as Record<string, unknown>)?.current ?? '—')}
                    </span>
                  </div>
                  <p className="pt-1 text-xs text-slate-500 dark:text-slate-400">{String(backupInv.note ?? '')}</p>
                </>
              ) : (
                <p className="text-sm text-slate-500 dark:text-slate-400">Inventory unavailable.</p>
              )}
            </div>
            <div className="text-sm space-y-2">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">Restore validation (non-destructive)</h3>
              {restoreCheck ? (
                <>
                  <div className="flex justify-between">
                    <span className="text-slate-500 dark:text-slate-400">Migration in sync</span>
                    <span className="font-medium">{restoreCheck.migration_in_sync ? 'yes' : 'no'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500 dark:text-slate-400">Authorization orphans</span>
                    <span className="font-medium text-slate-800 dark:text-slate-100">{String(restoreCheck.authorization_orphans)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500 dark:text-slate-400">Tenant data</span>
                    <span className="font-medium text-slate-800 dark:text-slate-100">
                      {String((restoreCheck.tenant_data as Record<string, number>)?.users ?? 0)} users · {(restoreCheck.tenant_data as Record<string, number>)?.workspaces ?? 0} ws · {(restoreCheck.tenant_data as Record<string, number>)?.organizations ?? 0} orgs
                    </span>
                  </div>
                  <div className="pt-1">
                    {badge(restoreCheck.restore_valid ? 'healthy' : 'degraded')}
                    <span className="ml-2 text-xs text-slate-500 dark:text-slate-400">restore integrity</span>
                  </div>
                </>
              ) : (
                <p className="text-sm text-slate-500 dark:text-slate-400">Restore validation unavailable.</p>
              )}
            </div>
          </div>
        </section>

        <h2 className="text-lg font-bold text-slate-900 dark:text-white mt-8 mb-3">Control plane (Phase 19)</h2>

        <div className="grid md:grid-cols-2 gap-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Global health</h2>
            {globalHealth ? (
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">API</dt>
                  <dd className="mt-1">{badge(String(globalHealth.api ?? 'unknown'))}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Database</dt>
                  <dd className="mt-1">{badge(String(globalHealth.database ?? 'unknown'))}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Broker</dt>
                  <dd className="mt-1">{badge(String(globalHealth.broker ?? 'unknown'))}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Workers</dt>
                  <dd className="mt-1">{badge(String(globalHealth.workers ?? 'unknown'))}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Providers</dt>
                  <dd className="mt-1">{badge(String(globalHealth.providers ?? 'unknown'))}</dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Scheduler leader</dt>
                  <dd className="mt-1 font-mono text-xs text-slate-800 dark:text-slate-100">
                    {schedulerLeader && (schedulerLeader.leader as Record<string, unknown> | null)
                      ? String((schedulerLeader.leader as Record<string, unknown>).leader_id)
                      : 'none elected'}
                  </dd>
                </div>
              </dl>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">Health unavailable.</p>
            )}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Control configuration snapshots</h2>
            {controlConfig ? (
              <div className="text-sm space-y-2">
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Active version</span>
                  <span className="font-semibold text-slate-800 dark:text-slate-100">
                    {controlConfig.active_version != null ? `v${String(controlConfig.active_version)}` : 'none'}
                  </span>
                </div>
                {(controlConfig.snapshots as Array<Record<string, unknown>> | undefined)?.slice(0, 5).map((s) => (
                  <div key={String(s.id)} className="flex justify-between text-xs">
                    <span className="text-slate-500 dark:text-slate-400 font-mono">v{String(s.version)} · {String(s.reason ?? 'no reason')}</span>
                    <span>{s.active ? badge('healthy') : badge('stopped')}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">No snapshots yet — create one via the control-plane API.</p>
            )}
          </section>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Regions</h2>
            {regions && regions.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                    <th className="pb-2">Region</th>
                    <th className="pb-2">Status</th>
                    <th className="pb-2">Health</th>
                    <th className="pb-2">Failover to</th>
                  </tr>
                </thead>
                <tbody>
                  {regions.map((r) => (
                    <tr key={String(r.region_id)} className="border-t border-slate-100 dark:border-slate-700">
                      <td className="py-2 font-medium text-slate-700 dark:text-slate-200">{String(r.region_id)}</td>
                      <td className="py-2">{badge(String(r.status).toLowerCase())}</td>
                      <td className="py-2 text-slate-600 dark:text-slate-300">{String(r.health_score)}</td>
                      <td className="py-2 text-slate-600 dark:text-slate-300">{r.failover_to ? String(r.failover_to) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">No regions registered — single-region deployment.</p>
            )}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Quarantined workers</h2>
            {quarantines && quarantines.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                    <th className="pb-2">Worker</th>
                    <th className="pb-2">Status</th>
                    <th className="pb-2">Failures</th>
                    <th className="pb-2">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {quarantines.map((q) => (
                    <tr key={String(q.worker_id)} className="border-t border-slate-100 dark:border-slate-700">
                      <td className="py-2 font-mono text-xs text-slate-700 dark:text-slate-200">{String(q.worker_id)}</td>
                      <td className="py-2">{badge(String(q.status).toLowerCase())}</td>
                      <td className="py-2 text-slate-600 dark:text-slate-300">{String(q.failure_count)}</td>
                      <td className="py-2 text-slate-600 dark:text-slate-300 text-xs">{q.reason ? String(q.reason) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">No quarantined workers.</p>
            )}
          </section>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Broker &amp; scheduler</h2>
            {broker2 ? (
              <div className="text-sm space-y-2">
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Backend</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">
                    {String((broker2.config as Record<string, unknown>)?.backend ?? '—')}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Delivery guarantee</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">
                    {String((broker2.guarantees as Record<string, unknown>)?.semantics ?? '—')}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Redis available</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">
                    {(broker2.redis_available as boolean) ? 'yes' : 'no (PostgreSQL broker)'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Recovery OK</span>
                  <span className="font-medium">
                    {badge((broker2.recovery as Record<string, unknown>)?.healthy ? 'healthy' : 'degraded')}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">Broker info unavailable.</p>
            )}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Provider capabilities</h2>
            {capabilities ? (
              <div className="text-sm space-y-2">
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Registered models</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">
                    {String((capabilities.registered as Array<unknown> | undefined)?.length ?? 0)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Routing modes</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">
                    cheapest · fastest · quality · balanced
                  </span>
                </div>
                <p className="pt-1 text-xs text-slate-500 dark:text-slate-400">
                  Routing respects capability, sensitivity, budget, and organization policy.
                </p>
              </div>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">Capability data unavailable.</p>
            )}
          </section>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Vector coverage &amp; drift</h2>
            {vectorCoverage ? (
              <div className="text-sm space-y-2">
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Total chunks</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">{String(vectorCoverage.total_chunks ?? '—')}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Embedded</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">{String(vectorCoverage.embedded_chunks ?? '—')}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Failed embeddings</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">{String(vectorCoverage.failed_embeddings ?? '—')}</span>
                </div>
                {vectorDrift && vectorDrift.stale_fraction != null && (
                  <div className="flex justify-between">
                    <span className="text-slate-500 dark:text-slate-400">Stale embedding fraction</span>
                    <span className="font-medium">
                      {badge(Number(vectorDrift.stale_fraction) > 0.2 ? 'degraded' : 'healthy')}
                      <span className="ml-2 text-xs text-slate-500 dark:text-slate-400">{(Number(vectorDrift.stale_fraction) * 100).toFixed(1)}%</span>
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">Vector coverage unavailable.</p>
            )}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Ingestion governor &amp; SLOs</h2>
            {ingestionGovernor ? (
              <div className="text-sm space-y-2 mb-4">
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Max file size</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">{String(ingestionGovernor.max_file_size_mb ?? '—')} MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Max pages</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">{String(ingestionGovernor.max_pages ?? '—')}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Extraction time</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">{String(ingestionGovernor.max_extraction_seconds ?? '—')}s</span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">Governor unavailable.</p>
            )}
            {slo19 ? (
              <div className="text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">SLO health</span>
                  <span>{badge(String((slo19.status ?? 'unknown')).toLowerCase())}</span>
                </div>
                {(slo19.items as Array<Record<string, unknown>> | undefined)?.slice(0, 4).map((s) => (
                  <div key={String(s.name)} className="flex justify-between text-xs mt-1">
                    <span className="text-slate-500 dark:text-slate-400">{String(s.name)}</span>
                    <span className="font-medium text-slate-800 dark:text-slate-100">{String(s.status)}</span>
                  </div>
                ))}
              </div>
            ) : null}
          </section>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Consistency &amp; DR readiness</h2>
            {consistency ? (
              <div className="text-sm space-y-2 mb-4">
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Checks run (dry)</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">{String(consistency.checks_run ?? '—')}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Issues</span>
                  <span className="font-medium">
                    {badge(Number(consistency.issues ?? 0) > 0 ? 'degraded' : 'healthy')}
                    <span className="ml-2 text-slate-800 dark:text-slate-100">{String(consistency.issues ?? 0)}</span>
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">Consistency check unavailable.</p>
            )}
            {drReadiness ? (
              <div className="text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">DR readiness score</span>
                  <span className="font-semibold text-slate-800 dark:text-slate-100">{String(drReadiness.score ?? '—')}/100</span>
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-slate-500 dark:text-slate-400">Backups recorded</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">{String(drReadiness.backup_count ?? '—')}</span>
                </div>
              </div>
            ) : null}
          </section>

          <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">AI quality &amp; search quality</h2>
            {evaluations && evaluations.length > 0 ? (
              <div className="text-sm space-y-1 mb-4">
                <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">RAG evaluations</p>
                {evaluations.slice(0, 4).map((e) => (
                  <div key={String(e.id)} className="flex justify-between text-xs">
                    <span className="text-slate-500 dark:text-slate-400 font-mono">{String(e.dataset ?? 'dataset')}</span>
                    <span>{e.passed ? badge('healthy') : badge('degraded')}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">No RAG evaluations recorded.</p>
            )}
            {searchQuality ? (
              <div className="text-sm space-y-1">
                <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Search quality</p>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500 dark:text-slate-400">Zero-result rate</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">
                    {searchQuality.zero_result_rate != null ? `${(Number(searchQuality.zero_result_rate) * 100).toFixed(1)}%` : '—'}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500 dark:text-slate-400">Avg latency</span>
                  <span className="font-medium text-slate-800 dark:text-slate-100">
                    {searchQuality.avg_latency_ms != null ? `${Number(searchQuality.avg_latency_ms)} ms` : '—'}
                  </span>
                </div>
              </div>
            ) : null}
          </section>
        </div>

        <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm mt-6">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Production cloud (Phase 22)</h2>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">Infrastructure capabilities</p>
              {p22infra ? (
                <div className="space-y-1">
                  {Object.entries(p22infra).filter(([, v]) => typeof v === 'boolean' || typeof v === 'string').slice(0, 6).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-xs">
                      <span className="text-slate-500 dark:text-slate-400">{k}</span>
                      <span className="font-medium text-slate-800 dark:text-slate-100 font-mono">{String(v)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-500 dark:text-slate-400">Unavailable.</p>
              )}
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">Vector platform</p>
              {p22backend ? (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500 dark:text-slate-400">Backend</span>
                    <span className="font-mono font-medium text-slate-800 dark:text-slate-100">{String((p22backend as Record<string, unknown>).backend ?? '—')}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500 dark:text-slate-400">Drift</span>
                    <span className="font-medium text-slate-800 dark:text-slate-100">{String((p22drift as Record<string, unknown> | null)?.drift_ratio ?? '—')}</span>
                  </div>
                </div>
              ) : (
                <p className="text-xs text-slate-500 dark:text-slate-400">Unavailable.</p>
              )}
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">Regions &amp; DR</p>
              {p22regions ? (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500 dark:text-slate-400">Registered regions</span>
                    <span className="font-medium text-slate-800 dark:text-slate-100">{Array.isArray((p22regions as Record<string, unknown>).items) ? ((p22regions as Record<string, unknown>).items as unknown[]).length : '—'}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500 dark:text-slate-400">Backup health</span>
                    <span className="font-medium text-slate-800 dark:text-slate-100">{String((p22dr as Record<string, unknown> | null)?.status ?? '—')}</span>
                  </div>
                </div>
              ) : (
                <p className="text-xs text-slate-500 dark:text-slate-400">Unavailable.</p>
              )}
            </div>
          </div>
          {p22streams ? (
            <div className="mt-4">
              <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-1">Operations stream</p>
              <div className="max-h-40 overflow-y-auto">
                {(Array.isArray((p22streams as Record<string, unknown>).items) ? ((p22streams as Record<string, unknown>).items as Array<Record<string, unknown>>) : []).slice(0, 10).map((e, i) => (
                  <div key={i} className="flex justify-between text-xs py-0.5 border-b border-slate-100 dark:border-slate-700/50">
                    <span className="text-slate-500 dark:text-slate-400 font-mono">{String(e.kind ?? e.stream ?? 'event')}</span>
                    <span className="text-slate-500 dark:text-slate-400">{String(e.created_at ?? '').slice(0, 19)}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </section>

        <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5 shadow-sm mt-6">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Global AI cloud (Phase 23)</h2>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">Global health</p>
              {p23health ? (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500 dark:text-slate-400">Status</span>
                    <span className={`font-medium font-mono ${String((p23health as Record<string, unknown>).status) === 'HEALTHY' ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}`}>{String((p23health as Record<string, unknown>).status ?? '—')}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500 dark:text-slate-400">Database</span>
                    <span className="font-mono font-medium text-slate-800 dark:text-slate-100">{String((p23health as Record<string, unknown>).database ?? '—')}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500 dark:text-slate-400">Broker</span>
                    <span className="font-mono font-medium text-slate-800 dark:text-slate-100">{String((p23health as Record<string, unknown>).broker ?? '—')}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500 dark:text-slate-400">Degraded</span>
                    <span className="font-mono font-medium text-slate-800 dark:text-slate-100">{Array.isArray((p23health as Record<string, unknown>).degraded_components) ? ((p23health as Record<string, unknown>).degraded_components as unknown[]).length : '—'}</span>
                  </div>
                </div>
              ) : (
                <p className="text-xs text-slate-500 dark:text-slate-400">Unavailable.</p>
              )}
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">Capability registry</p>
              {p23capabilities && p23capabilities.length > 0 ? (
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {p23capabilities.slice(0, 8).map((c, i) => (
                    <div key={i} className="flex justify-between text-xs">
                      <span className="text-slate-500 dark:text-slate-400">{String(c.component ?? '—')}</span>
                      <span className="font-mono font-medium text-slate-800 dark:text-slate-100">{String(c.state ?? '—')}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-500 dark:text-slate-400">Unavailable.</p>
              )}
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">Provider readiness</p>
              {p23providers ? (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500 dark:text-slate-400">Validated providers</span>
                    <span className="font-medium text-slate-800 dark:text-slate-100">{String((p23providers as Record<string, unknown>).count ?? '—')}</span>
                  </div>
                  {(Array.isArray((p23providers as Record<string, unknown>).items)
                    ? ((p23providers as Record<string, unknown>).items as Array<Record<string, unknown>>)
                    : []).slice(0, 4).map((p, i) => (
                    <div key={i} className="flex justify-between text-xs">
                      <span className="text-slate-500 dark:text-slate-400">{String(p.provider_kind ?? '—')}</span>
                      <span className="font-mono font-medium text-slate-800 dark:text-slate-100">{String(p.score ?? '—')}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-500 dark:text-slate-400">Unavailable.</p>
              )}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
