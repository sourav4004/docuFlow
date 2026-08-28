'use client';

import { RAGSource } from '@/lib/api';

interface SourceListProps {
  sources: RAGSource[];
}

export default function SourceList({ sources }: SourceListProps) {
  if (sources.length === 0) return null;

  return (
    <div className="px-6 py-4">
      <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-3">
        Sources
      </h4>
      <div className="space-y-2">
        {sources.map((source, index) => (
          <div
            key={`${source.chunk_id}-${index}`}
            className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900/40 rounded-lg border border-slate-100 dark:border-slate-700/50"
          >
            <div className="flex items-center space-x-3 min-w-0">
              {/* Document icon */}
              <div className="p-1.5 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded flex-shrink-0">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>

              <div className="min-w-0">
                <p className="text-sm font-medium text-slate-900 dark:text-white truncate">
                  {source.filename || `Document #${source.document_id}`}
                </p>
                <div className="flex items-center space-x-2 text-xs text-slate-500 dark:text-slate-400">
                  {source.page_start !== null && (
                    <span>
                      Page {source.page_start}
                      {source.page_end !== null && source.page_end !== source.page_start && (
                        <span>–{source.page_end}</span>
                      )}
                    </span>
                  )}
                  {source.page_start !== null && source.similarity_score !== null && (
                    <span>·</span>
                  )}
                  {source.similarity_score !== null && (
                    <span>{Math.round(source.similarity_score * 100)}% relevance</span>
                  )}
                </div>
              </div>
            </div>

            {/* Chunk index badge */}
            <span className="text-xs text-slate-400 dark:text-slate-500 font-mono flex-shrink-0 ml-2">
              chunk {source.chunk_index}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
