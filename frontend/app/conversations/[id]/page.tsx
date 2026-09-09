'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import api, { ConversationDetailResponse, ConversationMessage, PaginatedMessageResponse } from '@/lib/api';
import MessageBubble from '@/components/conversation/MessageBubble';
import PaginationControls from '@/components/conversation/PaginationControls';

const PAGE_SIZE = 20;

export default function ConversationPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const params = useParams();
  const conversationId = Number(params.id);

  const [conversation, setConversation] = useState<ConversationDetailResponse | null>(null);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [pagination, setPagination] = useState({ page: 1, total: 0, has_next: false, has_previous: false });
  const [loading, setLoading] = useState(true);
  const [pageLoading, setPageLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Title editing state
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleValue, setTitleValue] = useState('');
  const [titleSaving, setTitleSaving] = useState(false);
  const [titleError, setTitleError] = useState<string | null>(null);

  // Export state
  const [exporting, setExporting] = useState(false);

  const wasOnLastPageRef = useRef(true);

  // Message input state
  const [inputText, setInputText] = useState('');
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Streaming state
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState('');
  const [streamSources, setStreamSources] = useState<Array<{ document_id: number; filename: string | null; chunk_id: number; chunk_index: number; page_start: number | null; page_end: number | null; similarity_score: number | null; }>>([]);
  const [streamConfidence, setStreamConfidence] = useState<{ level: string; grounding_score: number } | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Message search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMode, setSearchMode] = useState(false);
  const [searchResults, setSearchResults] = useState<ConversationMessage[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchPagination, setSearchPagination] = useState({ page: 1, total: 0, has_next: false, has_previous: false });
  const searchRequestIdRef = useRef(0);

  // Fetch conversation metadata
  useEffect(() => {
    if (!user || !conversationId) return;
    let mounted = true;

    api.getConversation(conversationId)
      .then((data) => {
        if (mounted) setConversation(data);
      })
      .catch((err) => {
        if (mounted) setError(err instanceof Error ? err.message : 'Failed to load conversation');
      });

    return () => { mounted = false; };
  }, [user, conversationId]);

  // Track whether component is still mounted for async operations
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // Fetch messages for a given page
  const fetchMessages = useCallback(async (page: number, silent = false) => {
    if (!conversationId) return;
    if (!silent) setPageLoading(true);
    setError(null);

    try {
      const data: PaginatedMessageResponse = await api.getConversationMessages(conversationId, page, PAGE_SIZE);
      if (mountedRef.current) {
        setMessages(data.messages);
        setPagination({ page: data.page, total: data.total, has_next: data.has_next, has_previous: data.has_previous });
      }
    } catch (err) {
      if (!silent && mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to load messages');
      }
    } finally {
      if (mountedRef.current) {
        setPageLoading(false);
        setLoading(false);
      }
    }
  }, [conversationId]);

  // Load first page on mount
  useEffect(() => {
    if (user && conversationId) {
      fetchMessages(1);
    }
  }, [user, conversationId, fetchMessages]);

  // Handle page change
  const handlePageChange = useCallback((newPage: number) => {
    fetchMessages(newPage);
  }, [fetchMessages]);

  // Track whether the user is on the last page
  const updateLastPageTracking = useCallback(() => {
    const totalPages = Math.ceil(pagination.total / PAGE_SIZE);
    wasOnLastPageRef.current = pagination.page >= totalPages || pagination.total === 0;
  }, [pagination.page, pagination.total]);

  useEffect(() => {
    updateLastPageTracking();
  }, [updateLastPageTracking]);

  // Handle sending a message via streaming
  const handleSend = useCallback(async () => {
    const trimmed = inputText.trim();
    if (!trimmed || sending || !conversationId) return;

    setSending(true);
    setSendError(null);
    setStreaming(true);
    setStreamText('');
    setStreamSources([]);
    setStreamConfidence(null);

    const wasLastPage = wasOnLastPageRef.current;
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await api.sendMessageStream(
        conversationId,
        { content: trimmed },
        // onToken
        (text) => {
          if (mountedRef.current) {
            setStreamText((prev) => prev + text);
          }
        },
        // onSources
        (sources, confidence) => {
          if (mountedRef.current) {
            setStreamSources(sources);
            setStreamConfidence(confidence);
          }
        },
        // onComplete
        (_messageId, _grounded) => {
          if (mountedRef.current) {
            setInputText('');
            setStreaming(false);
            setStreamText('');
            setStreamSources([]);
            setStreamConfidence(null);
            // Refresh messages from server
            if (wasLastPage) {
              const totalAfter = pagination.total + 2;
              const lastPage = Math.ceil(totalAfter / PAGE_SIZE);
              fetchMessages(lastPage, true);
            } else {
              fetchMessages(pagination.page, true);
            }
          }
        },
        // onError
        (message) => {
          if (mountedRef.current) {
            setSendError(message);
            setStreaming(false);
            setStreamText('');
          }
        },
        controller.signal,
      );
    } catch (err) {
      if (mountedRef.current && !(err instanceof DOMException && err.name === 'AbortError')) {
        setSendError(err instanceof Error ? err.message : 'Failed to send message');
        setStreaming(false);
        setStreamText('');
      }
    } finally {
      if (mountedRef.current) {
        setSending(false);
        abortControllerRef.current = null;
      }
    }
  }, [inputText, sending, conversationId, pagination.total, pagination.page, fetchMessages]);

  // Auto-scroll to bottom on new messages or streaming text
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamText]);

  // Handle Enter key
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  // Cancel streaming
  const handleCancelStream = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    if (mountedRef.current) {
      setStreaming(false);
      setStreamText('');
      setSending(false);
    }
  }, []);

  // Title editing handlers
  const startEditTitle = useCallback(() => {
    if (!conversation) return;
    setTitleValue(conversation.title);
    setTitleError(null);
    setEditingTitle(true);
  }, [conversation]);

  const cancelEditTitle = useCallback(() => {
    setEditingTitle(false);
    setTitleError(null);
  }, []);

  const saveTitle = useCallback(async () => {
    const trimmed = titleValue.trim();
    if (!trimmed) {
      setTitleError('Title cannot be empty.');
      return;
    }
    if (trimmed.length > 200) {
      setTitleError('Title is too long (max 200 characters).');
      return;
    }
    if (!conversationId) return;

    setTitleSaving(true);
    setTitleError(null);
    try {
      const updated = await api.updateConversation(conversationId, trimmed);
      if (mountedRef.current) {
        setConversation((prev) => prev ? { ...prev, title: updated.title } : prev);
        setEditingTitle(false);
      }
    } catch (err) {
      if (mountedRef.current) {
        setTitleError(err instanceof Error ? err.message : 'Failed to update title');
      }
    } finally {
      if (mountedRef.current) {
        setTitleSaving(false);
      }
    }
  }, [titleValue, conversationId, conversation]);

  const handleTitleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      saveTitle();
    } else if (e.key === 'Escape') {
      cancelEditTitle();
    }
  }, [saveTitle, cancelEditTitle]);

  // Export handler
  const handleExport = useCallback(async () => {
    if (!conversationId || exporting) return;
    setExporting(true);
    try {
      await api.exportConversation(conversationId);
    } catch (err) {
      // Error is displayed by the API client
    } finally {
      if (mountedRef.current) {
        setExporting(false);
      }
    }
  }, [conversationId, exporting]);

  // Message search handler
  const handleSearch = useCallback(async (query: string, page = 1) => {
    if (!conversationId || !query.trim()) return;
    const requestId = ++searchRequestIdRef.current;
    setSearchLoading(true);
    try {
      const data = await api.searchMessages(conversationId, query.trim(), page, PAGE_SIZE);
      if (requestId === searchRequestIdRef.current && mountedRef.current) {
        setSearchResults(data.messages);
        setSearchPagination({ page: data.page, total: data.total, has_next: data.has_next, has_previous: data.has_previous });
        setSearchMode(true);
      }
    } catch (err) {
      if (requestId === searchRequestIdRef.current && mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to search messages');
      }
    } finally {
      if (requestId === searchRequestIdRef.current && mountedRef.current) {
        setSearchLoading(false);
      }
    }
  }, [conversationId]);

  const clearSearch = useCallback(() => {
    setSearchQuery('');
    setSearchMode(false);
    setSearchResults([]);
    setSearchPagination({ page: 1, total: 0, has_next: false, has_previous: false });
  }, []);

  useEffect(() => {
    const trimmed = searchQuery.trim();
    if (!trimmed) {
      clearSearch();
      return;
    }
    const debounce = setTimeout(() => handleSearch(trimmed), 300);
    return () => clearTimeout(debounce);
  }, [searchQuery, handleSearch, clearSearch]);

  // Auth redirect
  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  if (authLoading) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center">
        <div className="flex items-center space-x-3 text-slate-600 dark:text-slate-300">
          <div className="animate-spin h-6 w-6 border-3 border-blue-600 border-t-transparent rounded-full" />
          <span className="font-medium text-sm">Loading conversation...</span>
        </div>
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 h-14 flex items-center gap-3">
          <button
            onClick={() => router.push('/dashboard')}
            className="p-1.5 text-slate-500 hover:text-slate-700 dark:hover:text-slate-200 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            aria-label="Back to dashboard"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <div className="min-w-0 flex-1">
            {editingTitle ? (
              <div className="flex items-center gap-1.5">
                <input
                  type="text"
                  value={titleValue}
                  onChange={(e) => { setTitleValue(e.target.value); if (titleError) setTitleError(null); }}
                  onKeyDown={handleTitleKeyDown}
                  maxLength={200}
                  autoFocus
                  className="flex-1 min-w-0 px-2 py-0.5 text-sm font-bold bg-slate-50 dark:bg-slate-800 border border-blue-400 dark:border-blue-600 rounded-md text-slate-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
                <button
                  onClick={saveTitle}
                  disabled={titleSaving || !titleValue.trim()}
                  className="p-1 text-emerald-600 hover:text-emerald-700 dark:text-emerald-400 dark:hover:text-emerald-300 disabled:opacity-40 transition-colors"
                  aria-label="Save title"
                  title="Save"
                >
                  {titleSaving ? (
                    <div className="animate-spin h-3.5 w-3.5 border-2 border-emerald-600 border-t-transparent rounded-full" />
                  ) : (
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </button>
                <button
                  onClick={cancelEditTitle}
                  disabled={titleSaving}
                  className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 disabled:opacity-40 transition-colors"
                  aria-label="Cancel editing"
                  title="Cancel"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2 group">
                <h1 className="text-sm font-bold text-slate-900 dark:text-white truncate">
                  {conversation?.title || (loading ? 'Loading...' : 'Conversation')}
                </h1>
                {conversation && !loading && (
                  <button
                    onClick={startEditTitle}
                    className="p-0.5 text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-all"
                    aria-label="Edit title"
                    title="Edit title"
                  >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                    </svg>
                  </button>
                )}
              </div>
            )}
            {titleError && editingTitle && (
              <p className="text-[10px] text-red-600 dark:text-red-400 mt-0.5">{titleError}</p>
            )}
            {!editingTitle && pagination.total > 0 && (
              <p className="text-[10px] text-slate-500 dark:text-slate-400">
                {pagination.total} message{pagination.total !== 1 ? 's' : ''}
              </p>
            )}
          </div>
          {/* Actions */}
          {conversation && !loading && (
            <div className="flex items-center gap-1">
              {/* Search toggle */}
              <button
                onClick={() => { if (searchMode) clearSearch(); }}
                className={`p-1.5 rounded-lg transition-colors ${searchMode ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/40' : 'text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-950/40'}`}
                aria-label="Search messages"
                title="Search messages"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </button>
              {/* Export button */}
              <button
                onClick={handleExport}
                disabled={exporting}
                className="p-1.5 text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-950/40 rounded-lg transition-colors disabled:opacity-40"
                aria-label="Export conversation"
                title="Export as Markdown"
              >
              {exporting ? (
                <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full" />
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              )}
            </button>
            </div>
          )}
        </div>

        {/* Search bar */}
        {searchMode && (
          <div className="max-w-3xl mx-auto px-4 sm:px-6 py-2 border-t border-slate-200/80 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50">
            <div className="relative">
              <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search messages..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                autoFocus
                className="w-full pl-10 pr-9 py-1.5 text-sm bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={clearSearch}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">
              {searchLoading ? 'Searching...' : `${searchPagination.total} result${searchPagination.total !== 1 ? 's' : ''}`}
            </p>
          </div>
        )}
      </header>

      {/* Messages area */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-4">
          {loading ? (
            <div className="flex items-center justify-center py-16">
              <div className="flex items-center space-x-3 text-slate-500">
                <div className="animate-spin h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full" />
                <span className="text-sm">Loading messages...</span>
              </div>
            </div>
          ) : error ? (
            <div className="py-16 text-center">
              <div className="inline-flex items-center gap-2 px-4 py-3 bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 rounded-xl text-sm border border-red-200 dark:border-red-800">
                <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                {error}
              </div>
              <button
                onClick={() => { setError(null); setLoading(true); fetchMessages(pagination.page || 1); }}
                className="block mx-auto mt-3 text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
              >
                Try again
              </button>
            </div>
          ) : messages.length === 0 ? (
            <div className="py-16 text-center">
              <div className="inline-flex items-center justify-center w-12 h-12 bg-slate-100 dark:bg-slate-800 rounded-full mb-3">
                <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <p className="text-sm text-slate-500 dark:text-slate-400">No messages yet. Start the conversation below.</p>
            </div>
          ) : (
            <>
              {searchMode ? (
                /* Search results */
                <>
                  {searchLoading && (
                    <div className="flex items-center justify-center py-4">
                      <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full" />
                      <span className="ml-2 text-xs text-slate-500">Searching...</span>
                    </div>
                  )}

                  {searchResults.length > 0 ? (
                    <>
                      {searchPagination.total > PAGE_SIZE && (
                        <PaginationControls
                          page={searchPagination.page}
                          pageSize={PAGE_SIZE}
                          total={searchPagination.total}
                          hasPrevious={searchPagination.has_previous}
                          hasNext={searchPagination.has_next}
                          loading={searchLoading}
                          onPageChange={(p) => handleSearch(searchQuery, p)}
                        />
                      )}
                      <div className={searchLoading ? 'opacity-60 pointer-events-none' : ''}>
                        {searchResults.map((msg) => (
                          <MessageBubble key={msg.id} message={msg} />
                        ))}
                      </div>
                    </>
                  ) : !searchLoading ? (
                    <div className="py-16 text-center">
                      <p className="text-sm text-slate-500 dark:text-slate-400">No messages match your search.</p>
                    </div>
                  ) : null}
                </>
              ) : (
                /* Normal conversation messages */
                <>
                  {/* Page loading overlay */}
                  {pageLoading && (
                    <div className="flex items-center justify-center py-4">
                      <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full" />
                      <span className="ml-2 text-xs text-slate-500">Loading page...</span>
                    </div>
                  )}

                  {/* Top pagination */}
                  {pagination.total > PAGE_SIZE && (
                    <PaginationControls
                      page={pagination.page}
                      pageSize={PAGE_SIZE}
                      total={pagination.total}
                      hasPrevious={pagination.has_previous}
                      hasNext={pagination.has_next}
                      loading={pageLoading}
                      onPageChange={handlePageChange}
                    />
                  )}

                  {/* Messages */}
                  <div className={pageLoading ? 'opacity-60 pointer-events-none' : ''}>
                    {messages.map((msg) => (
                      <MessageBubble key={msg.id} message={msg} />
                    ))}
                  </div>

                  {/* Streaming assistant bubble */}
                  {streaming && (
                    <div className="flex gap-3 py-4">
                      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                        <div className="animate-spin h-3.5 w-3.5 border-2 border-blue-600 border-t-transparent rounded-full" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold text-slate-900 dark:text-white mb-1">DocuFlow</p>
                        <div className="px-4 py-3 bg-white dark:bg-slate-800 rounded-2xl rounded-tl-sm border border-slate-200/80 dark:border-slate-700/80 shadow-sm">
                          <p className="text-sm text-slate-800 dark:text-slate-200 whitespace-pre-wrap break-words">
                            {streamText || <span className="text-slate-400 dark:text-slate-500 italic">Generating...</span>}
                            <span className="inline-block w-1.5 h-4 bg-blue-500 ml-0.5 animate-pulse rounded-sm" />
                          </p>
                          {streamSources.length > 0 && (
                            <div className="mt-2 pt-2 border-t border-slate-100 dark:border-slate-700">
                              <p className="text-[10px] text-slate-500 dark:text-slate-400 mb-1">Sources:</p>
                              <div className="flex flex-wrap gap-1">
                                {streamSources.map((src, i) => (
                                  <span key={i} className="inline-flex items-center px-1.5 py-0.5 bg-slate-100 dark:bg-slate-700 rounded text-[10px] text-slate-600 dark:text-slate-300">
                                    {src.filename || `Document #${src.document_id}`}
                                    {src.page_start && ` p.${src.page_start}`}
                                  </span>
                                ))}
                              </div>
                              {streamConfidence && (
                                <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1">
                                  Confidence: {streamConfidence.level}
                                </p>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </>
              )}
            </>
          )}
        </div>
      </main>

      {/* Message input */}
      <footer className="sticky bottom-0 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-t border-slate-200/80 dark:border-slate-800">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-3">
          {sendError && (
            <div className="mb-2 px-3 py-2 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 rounded-lg text-xs text-red-700 dark:text-red-300">
              {sendError}
            </div>
          )}
          <div className="flex items-end gap-2">
            <textarea
              value={inputText}
              onChange={(e) => { setInputText(e.target.value); if (sendError) setSendError(null); }}
              onKeyDown={handleKeyDown}
              placeholder="Type a message..."
              disabled={sending}
              rows={1}
              maxLength={2000}
              className="flex-1 px-3 py-2 text-sm bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 disabled:opacity-50 resize-none transition-colors min-h-[38px] max-h-32"
              style={{ height: 'auto' }}
              onInput={(e) => {
                const target = e.currentTarget;
                target.style.height = 'auto';
                target.style.height = Math.min(target.scrollHeight, 128) + 'px';
              }}
            />
            {streaming ? (
              <button
                onClick={handleCancelStream}
                className="inline-flex items-center justify-center w-[38px] h-[38px] bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors flex-shrink-0"
                aria-label="Stop generation"
                title="Stop generation"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!inputText.trim() || sending}
                className="inline-flex items-center justify-center w-[38px] h-[38px] bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex-shrink-0"
                aria-label="Send message"
              >
                {sending ? (
                  <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                ) : (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
                  </svg>
                )}
              </button>
            )}
          </div>
          <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1">
            Press <kbd className="px-1 py-0.5 bg-slate-100 dark:bg-slate-700 rounded font-mono text-[9px]">Enter</kbd> to send, <kbd className="px-1 py-0.5 bg-slate-100 dark:bg-slate-700 rounded font-mono text-[9px]">Shift+Enter</kbd> for new line
          </p>
        </div>
      </footer>
    </div>
  );
}
