'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import api from '@/lib/api';

type Dict = Record<string, unknown>;

function asRecord(value: unknown): Dict {
  return (value && typeof value === 'object' ? value : {}) as Dict;
}

function asArray(value: unknown): Dict[] {
  return Array.isArray(value) ? (value as Dict[]) : [];
}

function str(value: unknown, fallback = '—'): string {
  if (value === null || value === undefined || value === '') return fallback;
  return String(value);
}

const levelColor: Record<string, string> = {
  OBSERVE: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
  RECOMMEND: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
  AUTO_LOW_RISK: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  AUTO_APPROVAL: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
  MANUAL_ONLY: 'bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300',
  ALLOWED: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  REQUIRES_APPROVAL: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
  BLOCKED: 'bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300',
  SUCCEEDED: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  FAILED: 'bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300',
  ESCALATED: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
  COOLDOWN_BLOCKED: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400',
  POLICY_BLOCKED: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400',
  HEALTHY: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  DEGRADED: 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300',
  UNHEALTHY: 'bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300',
  RECOVERING: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
  UNKNOWN: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400',
  SEV0: 'bg-red-200 text-red-900 dark:bg-red-900/60 dark:text-red-100',
  SEV1: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-200',
  SEV2: 'bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-200',
  SEV3: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200',
  SEV4: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
};

function chip(label: string) {
  const cls = levelColor[label] ?? 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300';
  return (
    <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>
      {label || '—'}
    </span>
  );
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <h2 className="text-sm font-semibold text-slate-900 dark:text-white">{title}</h2>
      {subtitle ? <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{subtitle}</p> : null}
      <div className="mt-3">{children}</div>
    </section>
  );
}

function Empty({ text }: { text: string }) {
  return <p className="text-xs text-slate-500 dark:text-slate-400">{text}</p>;
}

const LEVELS = ['OBSERVE', 'RECOMMEND', 'AUTO_LOW_RISK', 'AUTO_APPROVAL', 'MANUAL_ONLY'];

export default function AutonomyCenterPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const [workspaceId, setWorkspaceId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [policies, setPolicies] = useState<Dict[]>([]);
  const [operations, setOperations] = useState<Dict[]>([]);
  const [health, setHealth] = useState<Dict | null>(null);
  const [recoveries, setRecoveries] = useState<Dict[]>([]);
  const [diagnoses, setDiagnoses] = useState<Dict[]>([]);
  const [plans, setPlans] = useState<Dict[]>([]);
  const [candidates, setCandidates] = useState<Dict[]>([]);
  const [incidents, setIncidents] = useState<Dict[]>([]);
  const [activity, setActivity] = useState<Dict[]>([]);
  const [simOperation, setSimOperation] = useState('knowledge.recovery.embedding_gap');
  const [simResult, setSimResult] = useState<Dict | null>(null);
  const [stopReason, setStopReason] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!authLoading && !user) router.replace('/login');
  }, [authLoading, user, router]);

  useEffect(() => {
    const stored = typeof window !== 'undefined' ? window.localStorage.getItem('workspaceId') : null;
    if (stored) setWorkspaceId(Number(stored));
  }, []);

  const load = useCallback(async () => {
    if (!workspaceId) return;
    setLoading(true);
    setError(null);
    try {
      const [p, o, h, r, d, kpl, cand, inc, act] = await Promise.all([
        api.getAutonomyPolicies(workspaceId),
        api.getAutonomyOperations(workspaceId),
        api.getSystemHealthLatest(workspaceId),
        api.getRecoveryAttempts(workspaceId),
        api.getDiagnoses(workspaceId),
        api.getKnowledgeRecoveryPlans(workspaceId),
        api.getAdaptiveCandidates(workspaceId),
        api.getIncidentsP21(workspaceId),
        api.getActivityFeed(workspaceId),
      ]);
      setPolicies(asArray(asRecord(p).policies));
      setOperations(asArray(asRecord(o).operations));
      setHealth(asRecord(asRecord(h).snapshot));
      setRecoveries(asArray(asRecord(r).attempts));
      setDiagnoses(asArray(asRecord(d).reports));
      setPlans(asArray(asRecord(kpl).plans));
      setCandidates(asArray(asRecord(cand).candidates));
      setIncidents(asArray(asRecord(inc).incidents));
      setActivity(asArray(asRecord(act).items));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load autonomy center');
    } finally {
      setLoading(false);
    }
  }, [workspaceId]);

  useEffect(() => {
    if (workspaceId) void load();
  }, [workspaceId, load]);

  async function runSimulation() {
    if (!workspaceId) return;
    setBusy(true);
    try {
      setSimResult(asRecord(await api.simulateAutonomy(workspaceId, simOperation)));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Simulation failed');
    } finally {
      setBusy(false);
    }
  }

  async function changeLevel(policyId: number, newLevel: string) {
    if (!workspaceId) return;
    setBusy(true);
    try {
      await api.setAutonomyLevel(workspaceId, policyId, newLevel);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Level change rejected');
    } finally {
      setBusy(false);
    }
  }

  async function activateStop() {
    if (!workspaceId) return;
    setBusy(true);
    try {
      await api.emergencyStop(workspaceId, 'ALL', stopReason || 'operator requested');
      setStopReason('');
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Emergency stop failed');
    } finally {
      setBusy(false);
    }
  }

  if (authLoading || (!user && !authLoading)) {
    return <main className="p-6 text-sm text-slate-500">Loading…</main>;
  }

  const overall = str(health?.overall_state, 'UNKNOWN');

  return (
    <main className="mx-auto max-w-7xl p-6 text-slate-900 dark:text-slate-100">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Autonomy Center</h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Governed autonomous operations — policies, simulation, recovery, incidents.
            Every automatic action is policy-controlled, risk-classified, and audited.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link href="/ops" className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800">
            ← Operations
          </Link>
          <button
            type="button"
            onClick={load}
            disabled={loading || busy}
            className="rounded-lg bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50 dark:bg-white dark:text-slate-900"
          >
            {loading ? 'Loading…' : 'Refresh'}
          </button>
        </div>
      </header>

      {error ? (
        <p role="alert" className="mb-4 rounded-lg bg-rose-50 p-3 text-sm text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
          {error}
        </p>
      ) : null}

      {!workspaceId ? (
        <Card title="No workspace selected">
          <Empty text="Open a workspace from the Operations center to view its autonomy state." />
        </Card>
      ) : (
        <div className="space-y-6">
          {/* System health + emergency stop */}
          <div className="grid gap-4 md:grid-cols-3">
            <Card title="System health" subtitle="Aggregated across api, db, broker, workers, scheduler, providers, vector, ingestion, connectors, cache">
              <div className="flex items-center gap-2">
                {chip(overall)}
                <span className="text-xs text-slate-500">
                  unhealthy: {str(health?.unhealthy_count, '0')} · degraded: {str(health?.degraded_count, '0')}
                </span>
              </div>
            </Card>
            <Card title="Emergency stop" subtitle="Immediately vetoes all automatic execution for this workspace">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={stopReason}
                  onChange={(e) => setStopReason(e.target.value)}
                  placeholder="Reason (audited)"
                  aria-label="Emergency stop reason"
                  className="min-w-0 flex-1 rounded-lg border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-800"
                />
                <button
                  type="button"
                  onClick={activateStop}
                  disabled={busy}
                  className="rounded-lg bg-rose-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-rose-700 disabled:opacity-50"
                >
                  Stop
                </button>
              </div>
            </Card>
            <Card title="Autonomy simulation" subtitle="Zero side effects — would this operation auto-execute?">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={simOperation}
                  onChange={(e) => setSimOperation(e.target.value)}
                  aria-label="Operation type to simulate"
                  className="min-w-0 flex-1 rounded-lg border border-slate-300 px-2 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-800"
                />
                <button
                  type="button"
                  onClick={runSimulation}
                  disabled={busy}
                  className="rounded-lg bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50 dark:bg-white dark:text-slate-900"
                >
                  Simulate
                </button>
              </div>
              {simResult ? (
                <p className="mt-2 text-xs">
                  {chip(str(simResult.decision))}{' '}
                  <span className="text-slate-500">{str(simResult.reason, '')}</span>
                </p>
              ) : null}
            </Card>
          </div>

          {/* Autonomy policies */}
          <Card title="Autonomy policies" subtitle="OBSERVE → RECOMMEND → AUTO_LOW_RISK → AUTO_APPROVAL / MANUAL_ONLY. Level changes are server-validated and audited.">
            {policies.length === 0 ? (
              <Empty text="No policies yet — defaults to RECOMMEND (nothing executes automatically)." />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="text-xs uppercase tracking-wide text-slate-500">
                    <tr>
                      <th scope="col" className="py-2 pr-4">Operation</th>
                      <th scope="col" className="py-2 pr-4">Risk</th>
                      <th scope="col" className="py-2 pr-4">Level</th>
                      <th scope="col" className="py-2 pr-4">Budget</th>
                      <th scope="col" className="py-2">Change level</th>
                    </tr>
                  </thead>
                  <tbody>
                    {policies.map((p) => {
                      const id = Number(p.id);
                      return (
                        <tr key={id} className="border-t border-slate-100 dark:border-slate-800">
                          <td className="py-2 pr-4 font-mono text-xs">{str(p.operation_type)}</td>
                          <td className="py-2 pr-4 text-xs">{str(p.risk_level)}</td>
                          <td className="py-2 pr-4">{chip(str(p.autonomy_level))}</td>
                          <td className="py-2 pr-4 text-xs">{p.budget_limit_usd == null ? '—' : `$${str(p.budget_limit_usd)}`}</td>
                          <td className="py-2">
                            <select
                              aria-label={`Change level for ${str(p.operation_type)}`}
                              value={str(p.autonomy_level)}
                              disabled={busy}
                              onChange={(e) => changeLevel(id, e.target.value)}
                              className="rounded border border-slate-300 px-1.5 py-1 text-xs dark:border-slate-700 dark:bg-slate-800"
                            >
                              {LEVELS.map((l) => <option key={l} value={l}>{l}</option>)}
                            </select>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {/* Recovery + diagnosis */}
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Recovery attempts" subtitle="Idempotent, cooldown-aware auto-recovery; repeated failure escalates to an operator incident">
              {recoveries.length === 0 ? <Empty text="No recovery attempts recorded." /> : (
                <ul className="space-y-2 text-sm">
                  {recoveries.slice(0, 6).map((r) => (
                    <li key={String(r.id)} className="flex items-center justify-between gap-2">
                      <span className="truncate font-mono text-xs">{str(r.trigger)}</span>
                      {chip(str(r.status))}
                    </li>
                  ))}
                </ul>
              )}
            </Card>
            <Card title="Diagnoses" subtitle="Ranked hypotheses with evidence and confidence — never certainty without evidence">
              {diagnoses.length === 0 ? <Empty text="No diagnoses recorded." /> : (
                <ul className="space-y-2 text-sm">
                  {diagnoses.slice(0, 6).map((d) => (
                    <li key={String(d.id)}>
                      <span className="font-medium">{str(d.symptom)}</span>
                      <span className="text-slate-500"> → {str(d.top_cause)} ({str(d.top_confidence)}%)</span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>

          {/* Knowledge recovery + candidates */}
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Knowledge recovery plans" subtitle="Proposals only — execution requires an autonomy ALLOWED decision">
              {plans.length === 0 ? <Empty text="No recovery plans yet." /> : (
                <ul className="space-y-2 text-sm">
                  {plans.slice(0, 6).map((p) => (
                    <li key={String(p.id)} className="flex items-center justify-between gap-2">
                      <span className="truncate text-xs">{str(p.issue_kind)} → {str(p.target_type)}</span>
                      <span className="flex gap-1">{chip(str(p.risk_level))}{chip(str(p.decision || p.status))}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
            <Card title="Adaptive candidates" subtitle="Tuning candidates pass quality/regression/latency/cost/security gates, then require approval">
              {candidates.length === 0 ? <Empty text="No adaptation candidates yet." /> : (
                <ul className="space-y-2 text-sm">
                  {candidates.slice(0, 6).map((c) => (
                    <li key={String(c.id)} className="flex items-center justify-between gap-2">
                      <span className="truncate text-xs">{str(c.domain)} · {str(c.change_kind)}</span>
                      <span className="flex gap-1">{chip(str(c.status))}{c.promoted ? chip('ALLOWED') : null}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>

          {/* Incidents + operations audit */}
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Incidents" subtitle="SEV0–SEV4, auto-created from thresholds, with postmortem drafts pending human review">
              {incidents.length === 0 ? <Empty text="No incidents — all clear." /> : (
                <ul className="space-y-2 text-sm">
                  {incidents.slice(0, 6).map((i) => (
                    <li key={String(i.id)} className="flex items-center justify-between gap-2">
                      <span className="truncate">{str(i.title)}</span>
                      <span className="flex gap-1">{chip(str(i.severity))}{chip(str(i.status))}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
            <Card title="Autonomous operation audit" subtitle="Every decision — allowed, approval-required, or blocked — is recorded">
              {operations.length === 0 ? <Empty text="No autonomous operations yet." /> : (
                <ul className="space-y-2 text-sm">
                  {operations.slice(0, 8).map((o) => (
                    <li key={String(o.id)} className="flex items-center justify-between gap-2">
                      <span className="truncate font-mono text-xs">{str(o.operation_type)}</span>
                      <span className="flex items-center gap-1 text-xs text-slate-500">
                        {str(o.source)} {chip(str(o.decision))}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>

          {/* Personal AI activity */}
          <Card title="AI activity feed" subtitle="Suggestions, actions, decisions, approvals, recoveries — evidence and policy reasons only">
            {activity.length === 0 ? <Empty text="No activity items yet." /> : (
              <ul className="space-y-2 text-sm">
                {activity.slice(0, 8).map((a) => (
                  <li key={String(a.id)}>
                    {chip(str(a.item_kind))} <span className="ml-1">{str(a.title)}</span>
                    {a.explanation ? <span className="text-slate-500"> — {str(a.explanation)}</span> : null}
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      )}
    </main>
  );
}
