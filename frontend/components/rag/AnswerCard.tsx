'use client';

import { RAGResponse } from '@/lib/api';
import SourceList from './SourceList';

interface AnswerCardProps {
  response: RAGResponse;
}

export default function AnswerCard({ response }: AnswerCardProps) {
  const { answer, sources, grounded, retrieval_count } = response;

  return (
    <div className="bg-white dark:bg-slate-800/90 rounded-xl shadow-xs border border-slate-200/80 dark:border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-700/60">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className={`p-2 rounded-lg ${grounded ? 'bg-emerald-50 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400' : 'bg-amber-50 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400'}`}>
              {grounded ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
            </div>
            <div>
              <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                Answer
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {grounded ? 'Grounded in your documents' : 'Insufficient information found'}
              </p>
            </div>
          </div>

          {/* Source count badge */}
          {sources.length > 0 && (
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300">
              {sources.length} source{sources.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>

      {/* Answer Content */}
      <div className="px-6 py-5">
        {grounded ? (
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <p className="text-slate-800 dark:text-slate-200 whitespace-pre-wrap leading-relaxed">
              {answer}
            </p>
          </div>
        ) : (
          <div className="flex items-start space-x-3 p-4 bg-amber-50/80 dark:bg-amber-900/20 rounded-lg border border-amber-200/60 dark:border-amber-800/40">
            <svg className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
                {answer}
              </p>
              {retrieval_count === 0 && (
                <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                  No relevant information was found in your documents for this question.
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Sources */}
      {sources.length > 0 && (
        <div className="border-t border-slate-100 dark:border-slate-700/60">
          <SourceList sources={sources} />
        </div>
      )}
    </div>
  );
}
