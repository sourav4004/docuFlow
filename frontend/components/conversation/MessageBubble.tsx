'use client';

import { ConversationMessage } from '@/lib/api';

interface MessageBubbleProps {
  message: ConversationMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[85%] sm:max-w-[70%] rounded-xl px-4 py-3 ${
          isUser
            ? 'bg-blue-600 text-white'
            : 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-slate-100'
        }`}
      >
        {/* Role label */}
        <p
          className={`text-[10px] font-semibold uppercase tracking-wider mb-1 ${
            isUser
              ? 'text-blue-200'
              : 'text-slate-400 dark:text-slate-500'
          }`}
        >
          {isUser ? 'You' : 'DocuFlow'}
        </p>

        {/* Message content */}
        <p className="text-sm leading-relaxed whitespace-pre-wrap">
          {message.content}
        </p>

        {/* Sources for assistant messages */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2">
              Sources ({message.sources.length})
            </p>
            <div className="space-y-1.5">
              {message.sources.map((source) => (
                <div
                  key={source.id}
                  className="flex items-center text-xs text-slate-500 dark:text-slate-400"
                >
                  <span className="text-slate-400 dark:text-slate-500 mr-1.5">📄</span>
                  <span className="truncate">
                    {source.page_start !== null
                      ? `Page ${source.page_start}`
                      : `Chunk ${source.chunk_index}`}
                  </span>
                  {source.similarity_score !== null && (
                    <span className="ml-auto text-slate-400 dark:text-slate-500 flex-shrink-0 ml-2">
                      {Math.round(source.similarity_score * 100)}%
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Timestamp */}
        <p
          className={`text-[10px] mt-2 ${
            isUser
              ? 'text-blue-300'
              : 'text-slate-400 dark:text-slate-500'
          }`}
        >
          {new Date(message.created_at).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </p>
      </div>
    </div>
  );
}
