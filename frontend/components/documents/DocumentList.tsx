'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import api, { DocumentResponse } from '@/lib/api';
import { formatBytes, formatDate, statusLabel } from '@/lib/format';

interface DocumentListProps {
  documents: DocumentResponse[];
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
  onOpenUpload: () => void;
  onDeleteRequest: (document: DocumentResponse) => void;
}

// Formatting helpers live in lib/format.ts (shared app-wide).

export default function DocumentList({
  documents,
  loading,
  error,
  onRefresh,
  onOpenUpload,
  onDeleteRequest,
}: DocumentListProps) {
  const router = useRouter();
  const [activeActionDocId, setActiveActionDocId] = useState<number | null>(null);
  const [actionType, setActionType] = useState<'view' | 'download' | 'retry' | null>(null);
  const [actionError, setActionError] = useState<{ id: number; message: string } | null>(null);

  const handleView = async (doc: DocumentResponse) => {
    setActiveActionDocId(doc.id);
    setActionType('view');
    setActionError(null);

    try {
      const blob = await api.getDocumentBlob(doc.id);
      const blobUrl = URL.createObjectURL(blob);
      const newWindow = window.open(blobUrl, '_blank', 'noopener,noreferrer');
      if (!newWindow) {
        window.location.assign(blobUrl);
      }
      setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);
    } catch (err) {
      setActionError({
        id: doc.id,
        message: err instanceof Error ? err.message : 'Could not view document.',
      });
    } finally {
      setActiveActionDocId(null);
      setActionType(null);
    }
  };

  const handleDownload = async (doc: DocumentResponse) => {
    setActiveActionDocId(doc.id);
    setActionType('download');
    setActionError(null);

    try {
      const blob = await api.getDocumentBlob(doc.id);
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = doc.original_filename || `document_${doc.id}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 5000);
    } catch (err) {
      setActionError({
        id: doc.id,
        message: err instanceof Error ? err.message : 'Could not download document.',
      });
    } finally {
      setActiveActionDocId(null);
      setActionType(null);
    }
  };

  // 1. Loading Skeleton
  if (loading) {
    return (
      <div role="status" aria-live="polite" aria-busy="true" className="bg-white dark:bg-slate-800 rounded-xl shadow-xs border border-slate-200/80 dark:border-slate-700/80 p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="h-6 w-32 bg-slate-200 dark:bg-slate-700 rounded-md animate-pulse" />
          <div className="h-8 w-24 bg-slate-200 dark:bg-slate-700 rounded-md animate-pulse" />
        </div>
        <div className="space-y-3 pt-2">
          {[1, 2, 3].map((n) => (
            <div
              key={n}
              className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/40 rounded-lg animate-pulse"
            >
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-slate-200 dark:bg-slate-600 rounded-lg" />
                <div className="space-y-2">
                  <div className="h-4 w-48 bg-slate-200 dark:bg-slate-600 rounded-sm" />
                  <div className="h-3 w-28 bg-slate-200 dark:bg-slate-600 rounded-sm" />
                </div>
              </div>
              <div className="flex space-x-2">
                <div className="h-8 w-16 bg-slate-200 dark:bg-slate-600 rounded-md" />
                <div className="h-8 w-16 bg-slate-200 dark:bg-slate-600 rounded-md" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // 2. Error State
  if (error) {
    return (
      <div role="alert" aria-live="assertive" className="bg-white dark:bg-slate-800 rounded-xl shadow-xs border border-slate-200/80 dark:border-slate-700/80 p-8 text-center">
        <div className="w-12 h-12 mx-auto rounded-full bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 flex items-center justify-center mb-3">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-1">
          Unable to Load Documents
        </h3>
        <p className="text-sm text-slate-500 dark:text-slate-400 max-w-md mx-auto mb-5">
          {error}
        </p>
        <button
          onClick={onRefresh}
          className="inline-flex items-center space-x-2 px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 hover:bg-blue-100 dark:hover:bg-blue-900/50 rounded-lg transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          <span>Retry Loading</span>
        </button>
      </div>
    );
  }

  // 3. Empty State
  if (documents.length === 0) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-xs border border-slate-200/80 dark:border-slate-700/80 p-12 text-center">
        <div className="w-16 h-16 mx-auto rounded-2xl bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 flex items-center justify-center mb-4">
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">
          No documents yet
        </h3>
        <p className="text-slate-500 dark:text-slate-400 max-w-sm mx-auto mb-6 text-sm">
          Upload your first PDF document to get started with DocuFlow document operations.
        </p>
        <button
          onClick={onOpenUpload}
          className="inline-flex items-center space-x-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg shadow-xs transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          <span>Upload Document</span>
        </button>
      </div>
    );
  }

  const handleViewContent = (doc: DocumentResponse) => {
    router.push(`/documents/${doc.id}`);
  };

  const handleRetry = async (doc: DocumentResponse) => {
    setActiveActionDocId(doc.id);
    setActionType('retry');
    setActionError(null);

    try {
      await api.retryProcessing(doc.id);
      onRefresh();
    } catch (err) {
      setActionError({
        id: doc.id,
        message: err instanceof Error ? err.message : 'Could not retry processing.',
      });
    } finally {
      setActiveActionDocId(null);
      setActionType(null);
    }
  };

  function getStatusBadge(status: string) {
    switch (status) {
      case 'READY':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800/60">
            <span className="w-1.5 h-1.5 mr-1.5 rounded-full bg-emerald-500" />
            Ready
          </span>
        );
      case 'UPLOADED':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800/60">
            <span className="w-1.5 h-1.5 mr-1.5 rounded-full bg-blue-500" />
            Uploaded
          </span>
        );
      case 'QUEUED':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-800/60">
            <span className="w-1.5 h-1.5 mr-1.5 rounded-full bg-amber-500 animate-pulse" />
            Queued
          </span>
        );
      case 'PROCESSING':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800/60">
            <svg className="animate-spin h-3 w-3 mr-1.5 text-indigo-500" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Processing
          </span>
        );
      case 'FAILED':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-800/60">
            <span className="w-1.5 h-1.5 mr-1.5 rounded-full bg-red-500" />
            Failed
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300">
            {status}
          </span>
        );
    }
  }

  // 4. Populated State
  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-xs border border-slate-200/80 dark:border-slate-700/80 overflow-hidden">
      {/* Top action error banner if any */}
      {actionError && (
        <div className="p-3 bg-red-50 dark:bg-red-900/20 border-b border-red-200 dark:border-red-800/50 flex items-center justify-between text-xs text-red-700 dark:text-red-300">
          <span>Failed to process document #{actionError.id}: {actionError.message}</span>
          <button
            onClick={() => setActionError(null)}
            className="text-red-500 hover:text-red-700 font-bold ml-2"
          >
            ✕
          </button>
        </div>
      )}

      {/* Desktop & Tablet Table View */}
      <div className="hidden md:block overflow-x-auto">
        <table className="w-full text-left text-sm text-slate-600 dark:text-slate-300">
          <thead className="bg-slate-50 dark:bg-slate-700/50 text-xs uppercase font-semibold text-slate-500 dark:text-slate-400 border-b border-slate-200/80 dark:border-slate-700">
            <tr>
              <th scope="col" className="px-6 py-3.5">Document</th>
              <th scope="col" className="px-6 py-3.5">Type</th>
              <th scope="col" className="px-6 py-3.5">Size</th>
              <th scope="col" className="px-6 py-3.5">Status</th>
              <th scope="col" className="px-6 py-3.5">Uploaded</th>
              <th scope="col" className="px-6 py-3.5 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-700/60">
            {documents.map((doc) => {
              const isActionRunning = activeActionDocId === doc.id;

              return (
                <tr
                  key={doc.id}
                  className="hover:bg-slate-50/80 dark:hover:bg-slate-700/30 transition-colors group"
                >
                  {/* Filename & Icon */}
                  <td className="px-6 py-4">
                    <div className="flex items-center space-x-3 min-w-0">
                      <div className="p-2 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg flex-shrink-0">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                      </div>
                      <div className="min-w-0">
                        <p className="font-semibold text-slate-900 dark:text-white truncate max-w-xs xl:max-w-md" title={doc.original_filename}>
                          {doc.original_filename}
                        </p>
                        <p className="text-xs text-slate-400 dark:text-slate-500 font-mono">
                          ID: #{doc.id}
                        </p>
                      </div>
                    </div>
                  </td>

                  {/* File Type */}
                  <td className="px-6 py-4">
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 uppercase">
                      {doc.mime_type.split('/')[1] || 'PDF'}
                    </span>
                  </td>

                  {/* Size */}
                  <td className="px-6 py-4 font-mono text-xs text-slate-600 dark:text-slate-400 whitespace-nowrap">
                    {formatBytes(doc.file_size)}
                  </td>

                  {/* Status */}
                  <td className="px-6 py-4 whitespace-nowrap">
                    {getStatusBadge(doc.status)}
                  </td>

                  {/* Upload Date */}
                  <td className="px-6 py-4 text-xs text-slate-500 dark:text-slate-400 whitespace-nowrap">
                    {formatDate(doc.created_at)}
                  </td>

                  {/* Actions */}
                  <td className="px-6 py-4 text-right whitespace-nowrap">
                    <div className="inline-flex items-center space-x-1.5">
                      {/* View Action */}
                      <button
                        onClick={() => handleView(doc)}
                        disabled={isActionRunning}
                        className="inline-flex items-center px-2.5 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-700 hover:bg-slate-100 dark:hover:bg-slate-600 border border-slate-200 dark:border-slate-600 rounded-md transition-colors disabled:opacity-50"
                        title="View document in browser"
                      >
                        {isActionRunning && actionType === 'view' ? (
                          <svg className="animate-spin h-3.5 w-3.5 text-blue-600 dark:text-blue-400 mr-1" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                        ) : (
                          <svg className="w-3.5 h-3.5 mr-1 text-slate-500 dark:text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                          </svg>
                        )}
                        <span>View</span>
                      </button>

                      {/* Download Action */}
                      <button
                        onClick={() => handleDownload(doc)}
                        disabled={isActionRunning}
                        className="inline-flex items-center px-2.5 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-700 hover:bg-slate-100 dark:hover:bg-slate-600 border border-slate-200 dark:border-slate-600 rounded-md transition-colors disabled:opacity-50"
                        title="Download file"
                      >
                        {isActionRunning && actionType === 'download' ? (
                          <svg className="animate-spin h-3.5 w-3.5 text-blue-600 dark:text-blue-400 mr-1" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                        ) : (
                          <svg className="w-3.5 h-3.5 mr-1 text-slate-500 dark:text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                          </svg>
                        )}
                        <span>Download</span>
                      </button>

                      {/* View Content Action (READY only) */}
                      {doc.status === 'READY' && (
                        <button
                          onClick={() => handleViewContent(doc)}
                          disabled={isActionRunning}
                          className="inline-flex items-center px-2.5 py-1.5 text-xs font-medium text-purple-600 dark:text-purple-400 bg-white dark:bg-slate-700 hover:bg-purple-50 dark:hover:bg-purple-900/30 border border-slate-200 dark:border-slate-600 hover:border-purple-200 dark:hover:border-purple-800 rounded-md transition-colors disabled:opacity-50"
                          title="View extracted text"
                        >
                          <svg className="w-3.5 h-3.5 mr-1 text-purple-500 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          <span>Text</span>
                        </button>
                      )}

                      {/* Retry Action (FAILED or UPLOADED) */}
                      {(doc.status === 'FAILED' || doc.status === 'UPLOADED') && (
                        <button
                          onClick={() => handleRetry(doc)}
                          disabled={isActionRunning}
                          className="inline-flex items-center px-2.5 py-1.5 text-xs font-medium text-amber-600 dark:text-amber-400 bg-white dark:bg-slate-700 hover:bg-amber-50 dark:hover:bg-amber-900/30 border border-slate-200 dark:border-slate-600 hover:border-amber-200 dark:hover:border-amber-800 rounded-md transition-colors disabled:opacity-50"
                          title="Retry processing"
                        >
                          {isActionRunning && actionType === 'retry' ? (
                            <svg className="animate-spin h-3.5 w-3.5 text-amber-600 dark:text-amber-400 mr-1" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                          ) : (
                            <svg className="w-3.5 h-3.5 mr-1 text-amber-500 dark:text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                          )}
                          <span>Retry</span>
                        </button>
                      )}

                      {/* Delete Action */}
                      <button
                        onClick={() => onDeleteRequest(doc)}
                        disabled={isActionRunning}
                        className="inline-flex items-center px-2.5 py-1.5 text-xs font-medium text-red-600 dark:text-red-400 bg-white dark:bg-slate-700 hover:bg-red-50 dark:hover:bg-red-900/30 border border-slate-200 dark:border-slate-600 hover:border-red-200 dark:hover:border-red-800 rounded-md transition-colors disabled:opacity-50"
                        title="Delete document"
                      >
                        <svg className="w-3.5 h-3.5 mr-1 text-red-500 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                        <span>Delete</span>
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Mobile Card View */}
      <div className="md:hidden divide-y divide-slate-100 dark:divide-slate-700">
        {documents.map((doc) => {
          const isActionRunning = activeActionDocId === doc.id;

          return (
            <div key={doc.id} className="p-4 space-y-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-3 min-w-0">
                  <div className="p-2 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg flex-shrink-0">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <div className="min-w-0">
                    <p className="font-semibold text-slate-900 dark:text-white truncate">
                      {doc.original_filename}
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      {formatBytes(doc.file_size)} &bull; {formatDate(doc.created_at)}
                    </p>
                  </div>
                </div>
                {getStatusBadge(doc.status)}
              </div>

              {/* Action Buttons */}
              <div className="grid grid-cols-3 gap-2 pt-1">
                <button
                  onClick={() => handleView(doc)}
                  disabled={isActionRunning}
                  className="py-1.5 px-2 text-xs font-medium text-center text-slate-700 dark:text-slate-200 bg-slate-50 dark:bg-slate-700 hover:bg-slate-100 rounded-md border border-slate-200 dark:border-slate-600 disabled:opacity-50"
                >
                  View
                </button>
                <button
                  onClick={() => handleDownload(doc)}
                  disabled={isActionRunning}
                  className="py-1.5 px-2 text-xs font-medium text-center text-slate-700 dark:text-slate-200 bg-slate-50 dark:bg-slate-700 hover:bg-slate-100 rounded-md border border-slate-200 dark:border-slate-600 disabled:opacity-50"
                >
                  Download
                </button>
                <button
                  onClick={() => onDeleteRequest(doc)}
                  disabled={isActionRunning}
                  className="py-1.5 px-2 text-xs font-medium text-center text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 hover:bg-red-100 rounded-md border border-red-200 dark:border-red-800 disabled:opacity-50"
                >
                  Delete
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
