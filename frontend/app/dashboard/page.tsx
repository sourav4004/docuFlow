'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import api, { DocumentResponse } from '@/lib/api';
import DocumentUploadModal from '@/components/documents/DocumentUploadModal';
import DocumentList from '@/components/documents/DocumentList';
import DeleteConfirmModal from '@/components/documents/DeleteConfirmModal';

function formatBytes(bytes: number, decimals = 1): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

export default function DashboardPage() {
  const { user, loading: authLoading, logout } = useAuth();
  const router = useRouter();

  // Document state
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [loadingDocs, setLoadingDocs] = useState<boolean>(true);
  const [docsError, setDocsError] = useState<string | null>(null);

  // Modals state
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [documentToDelete, setDocumentToDelete] = useState<DocumentResponse | null>(null);

  // Toast / Status notification state
  const [bannerNotice, setBannerNotice] = useState<{
    type: 'success' | 'info' | 'error';
    text: string;
  } | null>(null);

  const fetchDocuments = useCallback(async () => {
    setLoadingDocs(true);
    setDocsError(null);
    try {
      const response = await api.listDocuments(100, 0);
      setDocuments(response.items || []);
    } catch (err) {
      setDocsError(err instanceof Error ? err.message : 'Failed to load documents');
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  // Authentication check
  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  // Load documents when user is authenticated
  useEffect(() => {
    let mounted = true;
    if (user) {
      api.listDocuments(100, 0)
        .then((response) => {
          if (mounted) {
            setDocuments(response.items || []);
            setLoadingDocs(false);
          }
        })
        .catch((err) => {
          if (mounted) {
            setDocsError(err instanceof Error ? err.message : 'Failed to load documents');
            setLoadingDocs(false);
          }
        });
    }
    return () => {
      mounted = false;
    };
  }, [user]);

  // Auto-dismiss banner notices
  useEffect(() => {
    if (bannerNotice) {
      const timer = setTimeout(() => setBannerNotice(null), 4500);
      return () => clearTimeout(timer);
    }
  }, [bannerNotice]);

  const handleLogout = async () => {
    try {
      await logout();
      router.push('/login');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const handleUploadSuccess = (uploadedDoc: DocumentResponse) => {
    // Add to list and refresh
    setDocuments((prev) => [uploadedDoc, ...prev.filter((d) => d.id !== uploadedDoc.id)]);
    setBannerNotice({
      type: 'success',
      text: `Document "${uploadedDoc.original_filename}" was uploaded and is ready.`,
    });
    // Trigger background full refresh to sync counters
    fetchDocuments();
  };

  const handleDeleteSuccess = (deletedId: number) => {
    setDocuments((prev) => prev.filter((d) => d.id !== deletedId));
    setBannerNotice({
      type: 'info',
      text: 'Document deleted successfully.',
    });
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

  if (!user) {
    return null;
  }

  // Calculate quick stats
  const totalDocuments = documents.length;
  const totalSizeBytes = documents.reduce((acc, doc) => acc + (doc.file_size || 0), 0);
  const readyCount = documents.filter((d) => d.status === 'READY').length;
  const processingCount = documents.filter((d) => d.status === 'PROCESSING' || d.status === 'QUEUED' || d.status === 'UPLOADED').length;
  const failedCount = documents.filter((d) => d.status === 'FAILED').length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-slate-100/70 to-slate-200/50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-800 text-slate-900 dark:text-slate-100">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-30 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo and Brand */}
            <div className="flex items-center space-x-3">
              <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-xs font-bold text-lg">
                D
              </div>
              <div>
                <span className="font-bold text-lg text-slate-900 dark:text-white tracking-tight">
                  DocuFlow
                </span>
                <span className="ml-2 px-2 py-0.5 text-2xs font-semibold uppercase tracking-wider bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 rounded-full">
                  Workspace
                </span>
              </div>
            </div>

            {/* User Profile & Actions */}
            <div className="flex items-center space-x-4">
              <div className="hidden sm:block text-right">
                <p className="text-sm font-semibold text-slate-900 dark:text-white leading-none">
                  {user.name}
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                  {user.email}
                </p>
              </div>

              <div className="h-6 w-px bg-slate-200 dark:bg-slate-700 hidden sm:block" />

              <button
                onClick={handleLogout}
                className="inline-flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/40 border border-slate-200 dark:border-slate-700 rounded-lg transition-colors"
                title="Sign out of DocuFlow"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Banner Notice Alert */}
        {bannerNotice && (
          <div
            className={`p-4 rounded-xl border flex items-center justify-between shadow-xs transition-all animate-in fade-in ${
              bannerNotice.type === 'success'
                ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200'
                : bannerNotice.type === 'error'
                ? 'bg-red-50 dark:bg-red-950/40 border-red-200 dark:border-red-800 text-red-800 dark:text-red-200'
                : 'bg-blue-50 dark:bg-blue-950/40 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-200'
            }`}
          >
            <div className="flex items-center space-x-2 text-sm font-medium">
              {bannerNotice.type === 'success' && (
                <svg className="w-5 h-5 flex-shrink-0 text-emerald-600 dark:text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              )}
              {bannerNotice.type === 'info' && (
                <svg className="w-5 h-5 flex-shrink-0 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              )}
              <span>{bannerNotice.text}</span>
            </div>
            <button
              onClick={() => setBannerNotice(null)}
              className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 p-1"
            >
              ✕
            </button>
          </div>
        )}

        {/* Dashboard Overview Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-slate-800/90 rounded-xl p-5 shadow-xs border border-slate-200/80 dark:border-slate-700">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Total Documents
              </span>
              <span className="p-2 bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-lg">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </span>
            </div>
            <div className="mt-3">
              <span className="text-2xl font-bold text-slate-900 dark:text-white">
                {loadingDocs ? '...' : totalDocuments}
              </span>
              <span className="text-xs text-slate-500 dark:text-slate-400 ml-2">PDF files</span>
            </div>
            {!loadingDocs && totalDocuments > 0 && (
              <div className="mt-2 flex items-center space-x-2 text-xs">
                <span className="text-emerald-600 dark:text-emerald-400">{readyCount} ready</span>
                {processingCount > 0 && <span className="text-amber-600 dark:text-amber-400">{processingCount} processing</span>}
                {failedCount > 0 && <span className="text-red-600 dark:text-red-400">{failedCount} failed</span>}
              </div>
            )}
          </div>

          <div className="bg-white dark:bg-slate-800/90 rounded-xl p-5 shadow-xs border border-slate-200/80 dark:border-slate-700">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Storage Used
              </span>
              <span className="p-2 bg-purple-50 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-lg">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                </svg>
              </span>
            </div>
            <div className="mt-3">
              <span className="text-2xl font-bold text-slate-900 dark:text-white">
                {loadingDocs ? '...' : formatBytes(totalSizeBytes)}
              </span>
              <span className="text-xs text-slate-500 dark:text-slate-400 ml-2">of 10 MB per file max</span>
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800/90 rounded-xl p-5 shadow-xs border border-slate-200/80 dark:border-slate-700">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                System Status
              </span>
              <span className="p-2 bg-emerald-50 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 rounded-lg">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </span>
            </div>
            <div className="mt-3 flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-sm font-semibold text-emerald-700 dark:text-emerald-400">
                Phase 3 Active
              </span>
              <span className="text-xs text-slate-500 dark:text-slate-400">&bull; Secure Vault</span>
            </div>
          </div>
        </div>

        {/* Document Management Section */}
        <section className="space-y-4">
          {/* Section Header with Actions */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-1 border-b border-slate-200/80 dark:border-slate-700">
            <div>
              <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white uppercase">
                DOCUMENTS
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Manage, view, download, and organize your uploaded PDF documents.
              </p>
            </div>

            <div className="flex items-center space-x-3">
              {/* Refresh Button */}
              <button
                onClick={fetchDocuments}
                disabled={loadingDocs}
                className="inline-flex items-center space-x-1.5 px-3 py-2 text-xs font-medium text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 rounded-lg shadow-2xs transition-colors disabled:opacity-50"
                title="Reload document list"
              >
                <svg
                  className={`w-3.5 h-3.5 ${loadingDocs ? 'animate-spin text-blue-600' : 'text-slate-500 dark:text-slate-400'}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span>Refresh</span>
              </button>

              {/* Upload Document CTA Button */}
              <button
                onClick={() => setIsUploadModalOpen(true)}
                className="inline-flex items-center space-x-1.5 px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 active:bg-blue-800 rounded-lg shadow-xs transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span>Upload Document</span>
              </button>
            </div>
          </div>

          {/* Document List Component */}
          <DocumentList
            documents={documents}
            loading={loadingDocs}
            error={docsError}
            onRefresh={fetchDocuments}
            onOpenUpload={() => setIsUploadModalOpen(true)}
            onDeleteRequest={(doc) => setDocumentToDelete(doc)}
          />
        </section>
      </main>

      {/* Upload Document Modal */}
      <DocumentUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUploadSuccess={handleUploadSuccess}
      />

      {/* Delete Confirmation Modal */}
      <DeleteConfirmModal
        isOpen={!!documentToDelete}
        document={documentToDelete}
        onClose={() => setDocumentToDelete(null)}
        onDeleteSuccess={handleDeleteSuccess}
      />
    </div>
  );
}
