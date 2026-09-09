'use client';

import { useCallback, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import api from '@/lib/api';

type Tab = 'api-keys' | 'webhooks' | 'usage' | 'integrations';

interface Workspace {
  id: number;
  name: string;
  description: string;
  role?: string;
}

interface ApiKeyItem {
  id: number;
  name: string;
  prefix: string;
  scopes: string[];
  last_used_at: string | null;
  expires_at: string | null;
  revoked_at: string | null;
  created_at: string;
}

interface WebhookItem {
  id: number;
  url: string;
  events: string[];
  status: string;
  description: string | null;
  created_at: string;
}

interface UsageData {
  usage: Record<string, number>;
  quota: Record<string, { used: number; limit: number | null; percent: number; remaining: number | null }>;
}

function formatPercent(pct: number): string {
  if (pct >= 100) return 'bg-red-500';
  if (pct >= 75) return 'bg-amber-500';
  return 'bg-emerald-500';
}

function formatDate(value: string | null | undefined): string {
  if (!value) return '—';
  return new Date(value).toLocaleDateString();
}

export default function SettingsPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  const [tab, setTab] = useState<Tab>('api-keys');
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [activeWorkspace, setActiveWorkspace] = useState<number | null>(null);

  // API keys state
  const [apiKeys, setApiKeys] = useState<ApiKeyItem[]>([]);
  const [keyName, setKeyName] = useState('');
  const [keyScopes, setKeyScopes] = useState<string[]>(['documents:read', 'search:read']);
  const [availableScopes, setAvailableScopes] = useState<string[]>([]);
  const [newKeySecret, setNewKeySecret] = useState<string | null>(null);
  const [loadingKeys, setLoadingKeys] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Webhooks state
  const [webhooks, setWebhooks] = useState<WebhookItem[]>([]);
  const [webhookUrl, setWebhookUrl] = useState('');
  const [webhookEvents, setWebhookEvents] = useState<string[]>(['*']);
  const [availableEvents, setAvailableEvents] = useState<string[]>([]);
  const [webhookSecret, setWebhookSecret] = useState<string | null>(null);
  const [loadingWebhooks, setLoadingWebhooks] = useState(false);

  // Usage state
  const [usage, setUsage] = useState<UsageData | null>(null);
  const [planInfo, setPlanInfo] = useState<{ plan: string; features: Record<string, boolean> } | null>(null);
  const [loadingUsage, setLoadingUsage] = useState(false);

  // Integrations state
  const [providers, setProviders] = useState<Array<{ provider: string; name: string; status: string; description: string }>>([]);

  const loadWorkspaces = useCallback(async () => {
    try {
      const resp = await api.listWorkspaces();
      setWorkspaces(resp.items || []);
      if (resp.items && resp.items.length > 0) {
        setActiveWorkspace((prev) => prev ?? resp.items![0].id);
      }
    } catch {
      setError('Failed to load workspaces');
    }
  }, []);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) {
      loadWorkspaces();
      api.listApiKeyScopes().then((r) => setAvailableScopes(r.scopes)).catch(() => {});
      api.listWebhookEvents().then((r) => setAvailableEvents(r.events)).catch(() => {});
      api.listIntegrationProviders().then((r) => setProviders(r.items || [])).catch(() => {});
    }
  }, [user, loadWorkspaces]);

  // Load data when active workspace changes
  useEffect(() => {
    if (!activeWorkspace) return;
    if (tab === 'api-keys') {
      setLoadingKeys(true);
      api.listApiKeys(activeWorkspace)
        .then((r) => setApiKeys(r.items || []))
        .catch(() => setError('Failed to load API keys'))
        .finally(() => setLoadingKeys(false));
    } else if (tab === 'webhooks') {
      setLoadingWebhooks(true);
      api.listWebhooks(activeWorkspace)
        .then((r) => setWebhooks(r.items || []))
        .catch(() => setError('Failed to load webhooks'))
        .finally(() => setLoadingWebhooks(false));
    } else if (tab === 'usage') {
      setLoadingUsage(true);
      api.getWorkspaceUsage(activeWorkspace)
        .then(setUsage)
        .catch(() => setError('Failed to load usage'))
        .finally(() => setLoadingUsage(false));
      api.getWorkspaceLimits(activeWorkspace)
        .then((r) => setPlanInfo({ plan: r.plan, features: r.features }))
        .catch(() => {});
    }
  }, [activeWorkspace, tab]);

  const handleCreateKey = async () => {
    if (!activeWorkspace || !keyName.trim()) return;
    setError(null);
    setNewKeySecret(null);
    try {
      const created = await api.createApiKey(activeWorkspace, keyName.trim(), keyScopes);
      setNewKeySecret(created.key || null);
      setKeyName('');
      const r = await api.listApiKeys(activeWorkspace);
      setApiKeys(r.items || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create API key');
    }
  };

  const handleRevokeKey = async (id: number) => {
    if (!activeWorkspace) return;
    try {
      await api.revokeApiKey(id);
      const r = await api.listApiKeys(activeWorkspace);
      setApiKeys(r.items || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to revoke API key');
    }
  };

  const handleCreateWebhook = async () => {
    if (!activeWorkspace || !webhookUrl.trim()) return;
    setError(null);
    setWebhookSecret(null);
    try {
      const created = await api.createWebhook(activeWorkspace, webhookUrl.trim(), webhookEvents);
      setWebhookSecret(created.signing_secret || null);
      setWebhookUrl('');
      const r = await api.listWebhooks(activeWorkspace);
      setWebhooks(r.items || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create webhook');
    }
  };

  const handleDeleteWebhook = async (id: number) => {
    if (!activeWorkspace) return;
    try {
      await api.deleteWebhook(id);
      const r = await api.listWebhooks(activeWorkspace);
      setWebhooks(r.items || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete webhook');
    }
  };

  const toggleScope = (scope: string) => {
    setKeyScopes((prev) =>
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope],
    );
  };

  const toggleEvent = (event: string) => {
    setWebhookEvents((prev) =>
      prev.includes(event) ? prev.filter((e) => e !== event) : [...prev, event],
    );
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center">
        <div className="flex items-center space-x-3 text-slate-600 dark:text-slate-300">
          <div className="animate-spin h-6 w-6 border-3 border-blue-600 border-t-transparent rounded-full" />
          <span className="font-medium text-sm">Loading DocuFlow...</span>
        </div>
      </div>
    );
  }

  if (!user) return null;

  const tabs: Array<{ key: Tab; label: string }> = [
    { key: 'api-keys', label: 'API Keys' },
    { key: 'webhooks', label: 'Webhooks' },
    { key: 'usage', label: 'Usage' },
    { key: 'integrations', label: 'Integrations' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-slate-100/70 to-slate-200/50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-800 text-slate-900 dark:text-slate-100">
      <header className="sticky top-0 z-30 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <a href="/dashboard" className="flex items-center space-x-3">
                <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center text-white font-bold text-lg">D</div>
                <div>
                  <span className="font-bold text-lg text-slate-900 dark:text-white tracking-tight">DocuFlow</span>
                  <span className="ml-2 px-2 py-0.5 text-2xs font-semibold uppercase tracking-wider bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-200 rounded-full">
                    Settings
                  </span>
                </div>
              </a>
            </div>
            <a
              href="/dashboard"
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 hover:text-blue-600 border border-slate-200 dark:border-slate-700 rounded-lg transition-colors"
            >
              <span>← Dashboard</span>
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Workspace selector */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Workspace Settings</h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">Manage developer access, notifications, and usage for the active workspace.</p>
          </div>
          <div className="flex items-center space-x-2">
            <label htmlFor="workspace-select" className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Workspace
            </label>
            <select
              id="workspace-select"
              value={activeWorkspace ?? ''}
              onChange={(e) => setActiveWorkspace(Number(e.target.value))}
              className="px-3 py-1.5 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {workspaces.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.name} ({w.role || 'member'})
                </option>
              ))}
            </select>
          </div>
        </div>

        {error && (
          <div className="p-3 rounded-xl border border-red-200 bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 text-sm flex items-center justify-between">
            <span>{error}</span>
            <button onClick={() => setError(null)} aria-label="Dismiss error" className="text-red-400 hover:text-red-600 p-1">✕</button>
          </div>
        )}

        {/* Tabs */}
        <nav className="flex space-x-1 border-b border-slate-200 dark:border-slate-700" aria-label="Settings sections">
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              aria-selected={tab === t.key}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg border-b-2 transition-colors ${
                tab === t.key
                  ? 'text-blue-600 dark:text-blue-400 border-blue-500'
                  : 'text-slate-500 dark:text-slate-400 border-transparent hover:text-slate-700 dark:hover:text-slate-200'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {/* API Keys tab */}
        {tab === 'api-keys' && (
          <section className="space-y-4" aria-label="API keys">
            <div className="bg-white dark:bg-slate-800/90 rounded-xl p-5 border border-slate-200/80 dark:border-slate-700">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">Create API Key</h2>
              <div className="mt-3 grid gap-3">
                <input
                  type="text"
                  value={keyName}
                  onChange={(e) => setKeyName(e.target.value)}
                  placeholder="Key name (e.g. CI pipeline)"
                  className="w-full px-3 py-2 text-sm bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  aria-label="API key name"
                />
                <fieldset>
                  <legend className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1.5">Scopes</legend>
                  <div className="flex flex-wrap gap-1.5">
                    {availableScopes.map((scope) => (
                      <button
                        key={scope}
                        type="button"
                        onClick={() => toggleScope(scope)}
                        aria-pressed={keyScopes.includes(scope)}
                        className={`px-2 py-1 text-xs rounded-md border transition-colors ${
                          keyScopes.includes(scope)
                            ? 'bg-blue-50 dark:bg-blue-950/50 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300'
                            : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300'
                        }`}
                      >
                        {scope}
                      </button>
                    ))}
                  </div>
                </fieldset>
                <div>
                  <button
                    onClick={handleCreateKey}
                    disabled={!keyName.trim() || loadingKeys}
                    className="px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50"
                  >
                    Create API Key
                  </button>
                </div>
              </div>
              {newKeySecret && (
                <div className="mt-4 p-3 rounded-lg border border-amber-300 bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-200 text-sm">
                  <p className="font-semibold">⚠ Copy this key now — it will never be shown again.</p>
                  <code className="block mt-2 px-3 py-2 rounded bg-white dark:bg-slate-900 border border-amber-200 dark:border-amber-800 break-all select-all">
                    {newKeySecret}
                  </code>
                  <button
                    onClick={() => { navigator.clipboard?.writeText(newKeySecret); }}
                    className="mt-2 px-3 py-1 text-xs font-medium text-amber-800 dark:text-amber-200 border border-amber-300 dark:border-amber-700 rounded-md hover:bg-amber-100 dark:hover:bg-amber-900/50"
                  >
                    Copy to clipboard
                  </button>
                </div>
              )}
            </div>

            <div className="bg-white dark:bg-slate-800/90 rounded-xl border border-slate-200/80 dark:border-slate-700 overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-200/80 dark:border-slate-700">
                <h2 className="text-sm font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Active Keys {loadingKeys && <span className="text-blue-500">(loading…)</span>}
                </h2>
              </div>
              {apiKeys.length === 0 ? (
                <p className="px-5 py-8 text-center text-sm text-slate-500 dark:text-slate-400">
                  No API keys yet. Create one above.
                </p>
              ) : (
                <ul className="divide-y divide-slate-200/80 dark:divide-slate-700">
                  {apiKeys.map((k) => (
                    <li key={k.id} className="px-5 py-3 flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-slate-900 dark:text-white">{k.name}</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400 truncate">
                          <code className="font-mono">{k.prefix}…</code> · {k.scopes.join(', ') || 'no scopes'} · created {formatDate(k.created_at)}
                          {k.revoked_at ? ' · revoked' : k.expires_at ? ` · expires ${formatDate(k.expires_at)}` : ''}
                        </p>
                      </div>
                      {!k.revoked_at && (
                        <button
                          onClick={() => handleRevokeKey(k.id)}
                          className="flex-shrink-0 px-3 py-1.5 text-xs font-medium text-red-600 dark:text-red-400 border border-red-200 dark:border-red-800 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/40"
                        >
                          Revoke
                        </button>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        )}

        {/* Webhooks tab */}
        {tab === 'webhooks' && (
          <section className="space-y-4" aria-label="Webhooks">
            <div className="bg-white dark:bg-slate-800/90 rounded-xl p-5 border border-slate-200/80 dark:border-slate-700">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">Add Webhook Endpoint</h2>
              <div className="mt-3 grid gap-3">
                <input
                  type="url"
                  value={webhookUrl}
                  onChange={(e) => setWebhookUrl(e.target.value)}
                  placeholder="https://example.com/hook"
                  className="w-full px-3 py-2 text-sm bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  aria-label="Webhook URL"
                />
                <fieldset>
                  <legend className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1.5">Events</legend>
                  <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto">
                    {availableEvents.map((evt) => (
                      <button
                        key={evt}
                        type="button"
                        onClick={() => toggleEvent(evt)}
                        aria-pressed={webhookEvents.includes(evt)}
                        className={`px-2 py-1 text-xs rounded-md border transition-colors ${
                          webhookEvents.includes(evt)
                            ? 'bg-blue-50 dark:bg-blue-950/50 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300'
                            : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300'
                        }`}
                      >
                        {evt}
                      </button>
                    ))}
                  </div>
                </fieldset>
                <div>
                  <button
                    onClick={handleCreateWebhook}
                    disabled={!webhookUrl.trim() || loadingWebhooks}
                    className="px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50"
                  >
                    Add Webhook
                  </button>
                </div>
              </div>
              {webhookSecret && (
                <div className="mt-4 p-3 rounded-lg border border-amber-300 bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-200 text-sm">
                  <p className="font-semibold">⚠ Signing secret — shown once. Use it to verify X-DocuFlow-Signature headers.</p>
                  <code className="block mt-2 px-3 py-2 rounded bg-white dark:bg-slate-900 border border-amber-200 dark:border-amber-800 break-all select-all">
                    {webhookSecret}
                  </code>
                </div>
              )}
            </div>

            <div className="bg-white dark:bg-slate-800/90 rounded-xl border border-slate-200/80 dark:border-slate-700 overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-200/80 dark:border-slate-700">
                <h2 className="text-sm font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Endpoints {loadingWebhooks && <span className="text-blue-500">(loading…)</span>}
                </h2>
              </div>
              {webhooks.length === 0 ? (
                <p className="px-5 py-8 text-center text-sm text-slate-500 dark:text-slate-400">
                  No webhook endpoints yet. Add one above.
                </p>
              ) : (
                <ul className="divide-y divide-slate-200/80 dark:divide-slate-700">
                  {webhooks.map((w) => (
                    <li key={w.id} className="px-5 py-3 flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-slate-900 dark:text-white truncate">{w.url}</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400 truncate">
                          {w.events.join(', ')} ·{' '}
                          <span className={w.status === 'ACTIVE' ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-500'}>
                            {w.status}
                          </span>
                        </p>
                      </div>
                      <button
                        onClick={() => handleDeleteWebhook(w.id)}
                        className="flex-shrink-0 px-3 py-1.5 text-xs font-medium text-red-600 dark:text-red-400 border border-red-200 dark:border-red-800 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/40"
                      >
                        Delete
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        )}

        {/* Usage tab */}
        {tab === 'usage' && (
          <section className="space-y-4" aria-label="Usage">
            {planInfo && (
              <div className="bg-white dark:bg-slate-800/90 rounded-xl p-5 border border-slate-200/80 dark:border-slate-700 flex items-center justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Current Plan</p>
                  <p className="text-xl font-bold text-slate-900 dark:text-white mt-1">{planInfo.plan}</p>
                </div>
                <div className="text-right text-xs text-slate-500 dark:text-slate-400">
                  {Object.entries(planInfo.features)
                    .filter(([, enabled]) => enabled)
                    .map(([feature]) => feature)
                    .join(', ') || 'No paid features'}
                </div>
              </div>
            )}
            {loadingUsage && <p className="text-sm text-slate-500 dark:text-slate-400">Loading usage…</p>}
            {usage && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {Object.entries(usage.quota).map(([metric, info]) => (
                  <div key={metric} className="bg-white dark:bg-slate-800/90 rounded-xl p-5 border border-slate-200/80 dark:border-slate-700">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        {metric.replace(/_/g, ' ')}
                      </span>
                      <span className="text-xs text-slate-500 dark:text-slate-400">
                        {info.used.toLocaleString()}
                        {info.limit !== null ? ` / ${info.limit.toLocaleString()}` : ''}
                      </span>
                    </div>
                    <div className="mt-3 h-2 rounded-full bg-slate-100 dark:bg-slate-700 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${formatPercent(info.percent)} transition-all`}
                        style={{ width: `${Math.min(info.percent, 100)}%` }}
                        role="progressbar"
                        aria-valuenow={Math.round(info.percent)}
                        aria-valuemin={0}
                        aria-valuemax={100}
                      />
                    </div>
                    <p className="mt-1.5 text-xs text-slate-500 dark:text-slate-400">
                      {info.limit === null ? 'Unlimited' : `${info.percent}% used`}
                    </p>
                  </div>
                ))}
              </div>
            )}
            {!loadingUsage && !usage && (
              <p className="text-sm text-slate-500 dark:text-slate-400">No usage data available.</p>
            )}
          </section>
        )}

        {/* Integrations tab */}
        {tab === 'integrations' && (
          <section className="space-y-4" aria-label="Integrations">
            <div className="bg-white dark:bg-slate-800/90 rounded-xl border border-slate-200/80 dark:border-slate-700 overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-200/80 dark:border-slate-700">
                <h2 className="text-sm font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">Integration Providers</h2>
              </div>
              <ul className="divide-y divide-slate-200/80 dark:divide-slate-700">
                {providers.map((p) => (
                  <li key={p.provider} className="px-5 py-3 flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-slate-900 dark:text-white">{p.name}</p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 truncate">{p.description}</p>
                    </div>
                    <span
                      className={`flex-shrink-0 px-2 py-1 text-2xs font-semibold uppercase tracking-wider rounded-full ${
                        p.status === 'AVAILABLE'
                          ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
                          : p.status === 'CONFIGURATION_REQUIRED'
                          ? 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300'
                          : 'bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-300'
                      }`}
                    >
                      {p.status.replace(/_/g, ' ')}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}