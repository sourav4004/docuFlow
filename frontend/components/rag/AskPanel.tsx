'use client';

import { useState, useCallback } from 'react';
import api, { DocumentResponse, RAGResponse } from '@/lib/api';

interface AskPanelProps {
  documents: DocumentResponse[];
  onAnswer: (response: RAGResponse) => void;
  onError: (error: string) => void;
  loading: boolean;
}

export default function AskPanel({ documents, onAnswer, onError, loading }: AskPanelProps) {
  const [question, setQuestion] = useState('');
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const readyDocuments = documents.filter((d) => d.status === 'READY');

  const validate = useCallback((): boolean => {
    const trimmed = question.trim();
    if (!trimmed) {
      setValidationError('Please enter a question.');
      return false;
    }
    if (trimmed.length > 2000) {
      setValidationError('Your question is too long (max 2000 characters).');
      return false;
    }
    setValidationError(null);
    return true;
  }, [question]);

  const handleSubmit = useCallback(async () => {
    if (!validate() || isSubmitting) return;

    setIsSubmitting(true);
    setValidationError(null);

    try {
      const response = await api.askQuestion({
        question: question.trim(),
        document_id: selectedDocId,
        top_k: 5,
      });
      onAnswer(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'An unexpected error occurred.';
      onError(message);
    } finally {
      setIsSubmitting(false);
    }
  }, [question, selectedDocId, validate, isSubmitting, onAnswer, onError]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  return (
    <div className="bg-white dark:bg-slate-800/90 rounded-xl shadow-xs border border-slate-200/80 dark:border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-700/60">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-lg">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">
              Ask Your Documents
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Get AI-powered answers grounded in your uploaded documents.
            </p>
          </div>
        </div>
      </div>

      {/* Form */}
      <div className="p-6 space-y-4">
        {/* Document Selector */}
        <div>
          <label
            htmlFor="doc-selector"
            className="block text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1.5"
          >
            Search Scope
          </label>
          <select
            id="doc-selector"
            value={selectedDocId ?? ''}
            onChange={(e) => {
              const val = e.target.value;
              setSelectedDocId(val ? Number(val) : null);
            }}
            disabled={isSubmitting || loading}
            className="w-full px-3 py-2 text-sm bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-600 rounded-lg text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 disabled:opacity-50 transition-colors"
          >
            <option value="">All Documents</option>
            {readyDocuments.map((doc) => (
              <option key={doc.id} value={doc.id}>
                {doc.original_filename}
              </option>
            ))}
          </select>
          {readyDocuments.length === 0 && !loading && (
            <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
              No processed documents available. Upload and process a PDF first.
            </p>
          )}
        </div>

        {/* Question Input */}
        <div>
          <label
            htmlFor="question-input"
            className="block text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1.5"
          >
            Question
          </label>
          <textarea
            id="question-input"
            rows={3}
            value={question}
            onChange={(e) => {
              setQuestion(e.target.value);
              if (validationError) setValidationError(null);
            }}
            onKeyDown={handleKeyDown}
            placeholder="What would you like to know about your documents?"
            disabled={isSubmitting}
            maxLength={2000}
            className="w-full px-3 py-2.5 text-sm bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-600 rounded-lg text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 disabled:opacity-50 resize-none transition-colors"
          />
          <div className="flex justify-between items-center mt-1">
            {validationError ? (
              <p className="text-xs text-red-600 dark:text-red-400">{validationError}</p>
            ) : (
              <span />
            )}
            <p className="text-xs text-slate-400 dark:text-slate-500">
              {question.length}/2000
            </p>
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex items-center justify-between">
          <p className="text-xs text-slate-400 dark:text-slate-500 hidden sm:block">
            Press <kbd className="px-1.5 py-0.5 bg-slate-100 dark:bg-slate-700 rounded text-[10px] font-mono">Ctrl</kbd>+<kbd className="px-1.5 py-0.5 bg-slate-100 dark:bg-slate-700 rounded text-[10px] font-mono">Enter</kbd> to submit
          </p>
          <button
            onClick={handleSubmit}
            disabled={isSubmitting || loading || !question.trim()}
            className="inline-flex items-center space-x-2 px-5 py-2.5 text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 active:bg-blue-800 rounded-lg shadow-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? (
              <>
                <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span>Searching &amp; Generating...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <span>Ask DocuFlow</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
