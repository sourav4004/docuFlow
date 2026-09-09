'use client';

interface PaginationControlsProps {
  page: number;
  pageSize: number;
  total: number;
  hasPrevious: boolean;
  hasNext: boolean;
  loading: boolean;
  onPageChange: (page: number) => void;
}

export default function PaginationControls({
  page,
  pageSize,
  total,
  hasPrevious,
  hasNext,
  loading,
  onPageChange,
}: PaginationControlsProps) {
  const totalPages = Math.ceil(total / pageSize);

  if (total === 0) return null;

  return (
    <div className="flex items-center justify-center gap-3 py-3">
      <button
        onClick={() => onPageChange(page - 1)}
        disabled={!hasPrevious || loading}
        className="inline-flex items-center gap-1 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        aria-label="Previous page"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Previous
      </button>

      <span className="text-xs font-medium text-slate-500 dark:text-slate-400 tabular-nums">
        Page {page} of {totalPages}
        <span className="text-slate-400 dark:text-slate-500 ml-1">({total} message{total !== 1 ? 's' : ''})</span>
      </span>

      <button
        onClick={() => onPageChange(page + 1)}
        disabled={!hasNext || loading}
        className="inline-flex items-center gap-1 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        aria-label="Next page"
      >
        Next
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>
    </div>
  );
}
