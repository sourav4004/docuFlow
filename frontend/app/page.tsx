'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import api, { HealthResponse } from '@/lib/api';
import { useAuth } from '@/lib/auth';

export default function Home() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [healthStatus, setHealthStatus] = useState<{
    loading: boolean;
    data: HealthResponse | null;
    error: string | null;
  }>({
    loading: true,
    data: null,
    error: null,
  });

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const data = await api.health();
        setHealthStatus({ loading: false, data, error: null });
      } catch (error) {
        setHealthStatus({
          loading: false,
          data: null,
          error: error instanceof Error ? error.message : 'Connection failed',
        });
      }
    };

    checkHealth();
  }, []);

  // Redirect to dashboard if already logged in
  useEffect(() => {
    if (!loading && user) {
      router.push('/dashboard');
    }
  }, [user, loading, router]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-16">
        <main className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-16">
            <h1 className="text-6xl font-bold text-slate-900 dark:text-white mb-4">
              DocuFlow
            </h1>
            <p className="text-xl text-slate-600 dark:text-slate-300 mb-8">
              AI-powered document operations
            </p>

            {!loading && !user && (
              <div className="flex gap-4 justify-center">
                <Link
                  href="/login"
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-6 rounded-md transition-colors"
                >
                  Sign In
                </Link>
                <Link
                  href="/register"
                  className="bg-white hover:bg-slate-50 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-900 dark:text-white font-medium py-3 px-6 rounded-md border border-slate-300 dark:border-slate-600 transition-colors"
                >
                  Create Account
                </Link>
              </div>
            )}
          </div>

          {/* Feature Cards */}
          <div className="grid md:grid-cols-3 gap-6 mb-16">
            <div className="bg-white dark:bg-slate-800 rounded-lg p-6 shadow-md">
              <div className="text-3xl mb-3">📄</div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                Document Management
              </h3>
              <p className="text-slate-600 dark:text-slate-300">
                Upload, store, and organize your documents securely
              </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-lg p-6 shadow-md">
              <div className="text-3xl mb-3">🤖</div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                AI Processing
              </h3>
              <p className="text-slate-600 dark:text-slate-300">
                Extract structured information using advanced AI
              </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-lg p-6 shadow-md">
              <div className="text-3xl mb-3">⚡</div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                Workflow Automation
              </h3>
              <p className="text-slate-600 dark:text-slate-300">
                Automate approvals, reminders, and deadlines
              </p>
            </div>
          </div>

          {/* Backend Connection Status */}
          <div className="bg-white dark:bg-slate-800 rounded-lg p-6 shadow-md">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
              System Status
            </h3>

            {healthStatus.loading && (
              <div className="flex items-center text-slate-600 dark:text-slate-300">
                <div className="animate-spin mr-2 h-5 w-5 border-2 border-slate-400 border-t-transparent rounded-full"></div>
                Checking backend connection...
              </div>
            )}

            {healthStatus.error && (
              <div className="flex items-center text-red-600 dark:text-red-400">
                <span className="mr-2">❌</span>
                Backend disconnected: {healthStatus.error}
              </div>
            )}

            {healthStatus.data && (
              <div className="space-y-2">
                <div className="flex items-center text-green-600 dark:text-green-400">
                  <span className="mr-2">✅</span>
                  Backend: {healthStatus.data.status}
                </div>
                <div className="flex items-center text-slate-600 dark:text-slate-300">
                  <span className="mr-2">💾</span>
                  Database: {healthStatus.data.database}
                </div>
              </div>
            )}
          </div>

          {/* Phase Notice */}
          <div className="mt-8 text-center text-sm text-slate-500 dark:text-slate-400">
            <p>Phase 3: Document Processing Foundation</p>
            <p className="mt-1">AI-powered features will be added in future phases</p>
          </div>
        </main>
      </div>
    </div>
  );
}
