'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import api from '@/lib/api';

interface Collection {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  document_count: number;
}

export default function CollectionsPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  const [collections, setCollections] = useState<Collection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Create form
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDescription, setNewDescription] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const fetchCollections = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listCollections(100, 0);
      setCollections(data.items || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load collections');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) fetchCollections();
  }, [user, fetchCollections]);

  const handleCreate = useCallback(async () => {
    if (!newName.trim()) {
      setCreateError('Collection name is required.');
      return;
    }
    setCreating(true);
    setCreateError(null);
    try {
      await api.createCollection(newName.trim(), newDescription.trim() || undefined);
      setNewName('');
      setNewDescription('');
      setShowCreate(false);
      fetchCollections();
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : 'Failed to create collection');
    } finally {
      setCreating(false);
    }
  }, [newName, newDescription, fetchCollections]);

  const handleDelete = useCallback(async (id: number, name: string) => {
    if (!confirm(`Delete collection "${name}"?`)) return;
    try {
      await api.deleteCollection(id);
      fetchCollections();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete collection');
    }
  }, [fetchCollections]);

  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center">
        <div className="animate-spin h-6 w-6 border-3 border-blue-600 border-t-transparent rounded-full" />
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      <header className="sticky top-0 z-30 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-14 flex items-center gap-3">
          <button onClick={() => router.push('/dashboard')} className="p-1.5 text-slate-500 hover:text-slate-700 dark:hover:text-slate-200 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
          </button>
          <h1 className="text-sm font-bold text-slate-900 dark:text-white">Collections</h1>
          <div className="flex-1" />
          <button onClick={() => setShowCreate(true)} className="inline-flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
            <span>New Collection</span>
          </button>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-6 space-y-4">
        {error && (
          <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm flex items-center justify-between">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600">✕</button>
          </div>
        )}

        {showCreate && (
          <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 space-y-3">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white">Create Collection</h3>
            {createError && <p className="text-xs text-red-600 dark:text-red-400">{createError}</p>}
            <input type="text" placeholder="Collection name" value={newName} onChange={(e) => setNewName(e.target.value)} className="w-full px-3 py-2 text-sm border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-900 text-slate-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:outline-none" autoFocus />
            <input type="text" placeholder="Description (optional)" value={newDescription} onChange={(e) => setNewDescription(e.target.value)} className="w-full px-3 py-2 text-sm border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-900 text-slate-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:outline-none" />
            <div className="flex items-center gap-2">
              <button onClick={handleCreate} disabled={creating || !newName.trim()} className="px-4 py-1.5 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg disabled:opacity-50 transition-colors">
                {creating ? 'Creating...' : 'Create'}
              </button>
              <button onClick={() => { setShowCreate(false); setCreateError(null); }} className="px-4 py-1.5 text-xs font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors">
                Cancel
              </button>
            </div>
          </div>
        )}

        {loading ? (
          <div className="text-center py-12 text-slate-500 dark:text-slate-400 text-sm">Loading collections...</div>
        ) : collections.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-400">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-400">No collections yet.</p>
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Create a collection to organize your documents.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {collections.map((col) => (
              <div key={col.id} className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 flex flex-col gap-2">
                <div className="flex items-start justify-between">
                  <h3 className="text-sm font-bold text-slate-900 dark:text-white truncate">{col.name}</h3>
                  <button onClick={() => handleDelete(col.id, col.name)} className="p-1 text-slate-400 hover:text-red-500 transition-colors flex-shrink-0" title="Delete collection">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                  </button>
                </div>
                {col.description && <p className="text-xs text-slate-500 dark:text-slate-400 line-clamp-2">{col.description}</p>}
                <div className="text-xs text-slate-400 dark:text-slate-500 mt-auto pt-2 border-t border-slate-100 dark:border-slate-700">
                  {col.document_count} document{col.document_count !== 1 ? 's' : ''}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
