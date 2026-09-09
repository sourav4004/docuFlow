'use client';

// ============================================================
// DocuFlow shared UI primitives (Phase 24, Steps 36 / 85-87 / 46)
//
// One design system for status, loading, error and empty states,
// used by dashboard, documents, conversations, /ops and
// /ai-control. Standard Tailwind tokens; accessible by default
// (roles, aria-live announcements, focus-visible rings).
// ============================================================

import { ReactNode } from 'react';

// ------------------------------------------------------------
// StatusBadge — honest status vocabulary, never color-only
// ------------------------------------------------------------

export type StatusTone =
  | 'ok'        // healthy / ready / real
  | 'warn'      // degraded / fallback
  | 'error'     // failed / unavailable
  | 'info'      // neutral / simulated
  | 'muted';    // unknown / pending

const TONE_STYLES: Record<StatusTone, string> = {
  ok: 'bg-green-100 text-green-800 border-green-200',
  warn: 'bg-amber-100 text-amber-800 border-amber-200',
  error: 'bg-red-100 text-red-800 border-red-200',
  info: 'bg-blue-100 text-blue-800 border-blue-200',
  muted: 'bg-gray-100 text-gray-600 border-gray-200',
};

export function StatusBadge({
  tone,
  label,
  title,
}: {
  tone: StatusTone;
  label: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-xs font-medium ${TONE_STYLES[tone]}`}
    >
      {/* Shape + text, so status never relies on color alone */}
      <span aria-hidden="true" className="text-[10px] leading-none">
        {tone === 'ok' ? '●' : tone === 'warn' ? '▲' : tone === 'error' ? '✕' : tone === 'info' ? '◆' : '○'}
      </span>
      {label}
    </span>
  );
}

// ------------------------------------------------------------
// Loading — skeleton + spinner + announced region
// ------------------------------------------------------------

export function Skeleton({ className = '' }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse rounded bg-gray-200 ${className}`}
    />
  );
}

export function Loading({
  label = 'Loading…',
  className = '',
}: {
  label?: string;
  className?: string;
}) {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-busy="true"
      className={`flex items-center justify-center gap-2 py-8 text-sm text-gray-500 ${className}`}
    >
      <svg
        className="h-4 w-4 animate-spin text-gray-400"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
      </svg>
      {label}
    </div>
  );
}

export function SkeletonCard({ lines = 3 }: { lines?: number }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <Skeleton className="mb-3 h-4 w-1/3" />
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className="mb-2 h-3 w-full" />
      ))}
    </div>
  );
}

// ------------------------------------------------------------
// ErrorState — user-friendly failure with optional retry
// ------------------------------------------------------------

export function ErrorState({
  message,
  onRetry,
  title = 'Something went wrong',
}: {
  message: string;
  onRetry?: () => void;
  title?: string;
}) {
  return (
    <div
      role="alert"
      aria-live="assertive"
      className="rounded-lg border border-red-200 bg-red-50 p-4"
    >
      <div className="flex items-start gap-2">
        <span aria-hidden="true" className="mt-0.5 text-red-500">✕</span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-red-800">{title}</p>
          <p className="mt-0.5 break-words text-sm text-red-700">{message}</p>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="focus-visible:ring-2 focus-visible:ring-red-500 mt-2 rounded border border-red-300 bg-white px-3 py-1 text-sm font-medium text-red-800 hover:bg-red-100 focus:outline-none"
            >
              Retry
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------
// EmptyState — explains what the user can do next
// ------------------------------------------------------------

export function EmptyState({
  title,
  hint,
  action,
}: {
  title: string;
  hint?: string;
  action?: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-dashed border-gray-300 bg-gray-50 p-8 text-center">
      <p className="text-sm font-medium text-gray-700">{title}</p>
      {hint && <p className="mt-1 text-sm text-gray-500">{hint}</p>}
      {action && <div className="mt-3 flex justify-center">{action}</div>}
    </div>
  );
}

// ------------------------------------------------------------
// Card / Panel — consistent container styling
// ------------------------------------------------------------

export function Card({
  title,
  actions,
  children,
  className = '',
}: {
  title?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-lg border border-gray-200 bg-white shadow-sm ${className}`}
    >
      {(title || actions) && (
        <header className="flex items-center justify-between border-b border-gray-100 px-4 py-3">
          {title && (
            <h2 className="text-sm font-semibold text-gray-900">{title}</h2>
          )}
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

// ------------------------------------------------------------
// Button — primary / secondary / danger variants
// ------------------------------------------------------------

const BUTTON_VARIANTS = {
  primary:
    'bg-blue-600 text-white hover:bg-blue-700 focus-visible:ring-blue-500 disabled:bg-blue-300',
  secondary:
    'border border-gray-300 bg-white text-gray-800 hover:bg-gray-50 focus-visible:ring-gray-400 disabled:text-gray-400',
  danger:
    'bg-red-600 text-white hover:bg-red-700 focus-visible:ring-red-500 disabled:bg-red-300',
} as const;

export function Button({
  variant = 'primary',
  className = '',
  children,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: keyof typeof BUTTON_VARIANTS;
}) {
  return (
    <button
      className={`focus-visible:ring-2 rounded px-3 py-1.5 text-sm font-medium transition focus:outline-none disabled:cursor-not-allowed ${BUTTON_VARIANTS[variant]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}
