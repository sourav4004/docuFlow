'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import api, { ConversationResponse } from '@/lib/api';

export default function ConversationsPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  const [conversations, setConversations] = useState<ConversationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [search, setSearch] = useState('');
  const [searchLoading, setSearchLoading] = useState(false);
  const searchRequestIdRef = useRef(0);

  const fetchConversations = useCallback(async (searchQuery?: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listConversations(50, 0, searchQuery);
      setConversations(data.items || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load conversations');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSearch = useCallback(async (query: string) => {
    const requestId = ++searchRequestIdRef.current;
    setSearchLoading(true);
    try {
      const data = await api.listConversations(50, 0, query);
      if (requestId === searchRequestIdRef.current) {
        setConversations(data.items || []);
        setError(null);
      }
    } catch (err) {
      if (requestId === searchRequestIdRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to search conversations');
      }
    } finally {
      if (requestId === searchRequestIdRef.current) {
        setSearchLoading(false);
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    const trimmed = search.trim();
    if (!trimmed) {
      fetchConversations();
      return;
    }
    const debounce = setTimeout(() => handleSearch(trimmed), 300);
    return () => clearTimeout(debounce);
  }, [search, fetchConversations, handleSearch]);

  // Clear delete errors after a delay
  useEffect(() => {
    if (error && !loading) {
      const timer = setTimeout(() => setError(null), 6000);
      return () => clearTimeout(timer);
    }
  }, [error, loading]);

  useEffect(() => {
    if (user) fetchConversations();
  }, [user, fetchConversations]);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  const handleNewConversation = useCallback(async () => {
    setCreating(true);
    try {
      const conv = await api.createConversation({ title: 'New Conversation' });
      router.push(`/conversations/${conv.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create conversation');
      setCreating(false);
    }
  }, [router]);

  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center">
        <div className="flex items-center space-x-3 text-slate-600 dark:text-slate-300">
          <div className="animate-spin h-6 w-6 border-3 border-blue-600 border-t-transparent rounded-full" />
          <span className="font-medium text-sm">Loading...</span>
        </div>
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => router.push('/dashboard')}
              className="p-1.5 text-slate-500 hover:text-slate-700 dark:hover:text-slate-200 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              aria-label="Back to dashboard"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <h1 className="text-sm font-bold text-slate-900 dark:text-white">Conversations</h1>
          </div>
          <button
            onClick={handleNewConversation}
            disabled={creating}
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-xs transition-colors disabled:opacity-50"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>New Conversation</span>
          </button>
        </div>
      </header>

      {/* Search */}
      <div className="max-w-3xl mx-auto px-4 sm:px-6 pt-4">
        <div className="relative">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            placeholder="Search conversations..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-9 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <main className="max-w-3xl mx-auto px-4 sm:px-6 py-6">
        {(loading && !searchLoading) || searchLoading ? (
          <div className="flex items-center justify-center py-16">
            <div className="flex items-center space-x-3 text-slate-500">
              <div className="animate-spin h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full" />
              <span className="text-sm">Loading conversations...</span>
            </div>
          </div>
        ) : error ? (
          <div className="py-16 text-center">
            <div className="inline-flex items-center gap-2 px-4 py-3 bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 rounded-xl text-sm border border-red-200 dark:border-red-800">
              {error}
            </div>
            <button
              onClick={() => fetchConversations(search.trim() || undefined)}
              className="block mx-auto mt-3 text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
            >
              Try again
            </button>
          </div>
        ) : conversations.length === 0 ? (
          <div className="py-16 text-center">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-slate-100 dark:bg-slate-800 rounded-full mb-3">
              <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">
              {search.trim() ? 'No conversations found' : 'No conversations yet.'}
            </p>
            {!search.trim() && (
              <button
                onClick={handleNewConversation}
                disabled={creating}
                className="inline-flex items-center space-x-1.5 px-4 py-2 text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-xs transition-colors disabled:opacity-50"
              >
                Start your first conversation
              </button>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className="group flex items-center gap-3 p-4 bg-white dark:bg-slate-800/90 rounded-xl border border-slate-200/80 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-700 hover:shadow-sm transition-all"
              >
                <button
                  onClick={() => router.push(`/conversations/${conv.id}`)}
                  className="flex-1 text-left min-w-0"
                >
                  <h3 className="text-sm font-semibold text-slate-900 dark:text-white truncate">
                    {conv.title}
                  </h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    {new Date(conv.updated_at).toLocaleDateString(undefined, {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric',
                    })}
                  </p>
                </button>
                <button
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (!window.confirm(`Delete "${conv.title}"?`)) return;
                    setDeletingId(conv.id);
                    try {
                      await api.deleteConversation(conv.id);
                      setConversations((prev) => prev.filter((c) => c.id !== conv.id));
                    } catch (err) {
                      setError(err instanceof Error ? err.message : 'Failed to delete conversation');
                    } finally {
                      setDeletingId(null);
                    }
                  }}
                  disabled={deletingId === conv.id}
                  className="p-1.5 text-slate-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/40 rounded-lg opacity-0 group-hover:opacity-100 focus:opacity-100 transition-all disabled:opacity-50"
                  aria-label={`Delete conversation ${conv.title}`}
                  title="Delete conversation"
                >
                  {deletingId === conv.id ? (
                    <div className="animate-spin h-4 w-4 border-2 border-red-600 border-t-transparent rounded-full" />
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  )}
                </button>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
