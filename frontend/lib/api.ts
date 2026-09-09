const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface HealthResponse {
  status: string;
  database: string;
}

export interface UserResponse {
  id: number;
  name: string;
  email: string;
  created_at: string;
  updated_at: string;
}

export interface RegisterData {
  name: string;
  email: string;
  password: string;
}

export interface LoginData {
  email: string;
  password: string;
}

export interface DocumentResponse {
  id: number;
  original_filename: string;
  mime_type: string;
  file_size: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  items: DocumentResponse[];
  total: number;
  limit: number;
  offset: number;
}

export interface DeleteResponse {
  message: string;
}

// RAG types
export interface RAGSource {
  document_id: number;
  filename: string | null;
  chunk_id: number;
  chunk_index: number;
  page_start: number | null;
  page_end: number | null;
  similarity_score: number | null;
}

export interface RAGRequest {
  question: string;
  document_id?: number | null;
  top_k?: number;
}

export interface RAGResponse {
  answer: string;
  sources: RAGSource[];
  grounded: boolean;
  retrieval_count: number;
  model: string | null;
  provider: string | null;
}

export interface DocumentStatusResponse {
  document_id: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface DocumentContentResponse {
  document_id: number;
  status: string;
  extracted_text: string | null;
  error_message: string | null;
}

// Conversation types
export interface ConversationResponse {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationListResponse {
  items: ConversationResponse[];
  total: number;
  limit: number;
  offset: number;
}

export interface ConversationDetailResponse {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ConversationMessage[];
}

export interface ConversationMessage {
  id: number;
  role: string;
  content: string;
  created_at: string;
  sources: ConversationSource[];
}

export interface ConversationSource {
  id: number;
  document_id: number;
  chunk_id: number;
  chunk_index: number;
  page_start: number | null;
  page_end: number | null;
  similarity_score: number | null;
  created_at: string;
}

export interface PaginatedMessageResponse {
  messages: ConversationMessage[];
  page: number;
  page_size: number;
  total: number;
  has_next: boolean;
  has_previous: boolean;
}

export interface SendMessageRequest {
  content: string;
  document_id?: number | null;
}

export interface SendMessageResponse {
  user_message: ConversationMessage;
  assistant_message: ConversationMessage;
  sources: Array<{
    document_id: number;
    filename: string | null;
    chunk_id: number;
    chunk_index: number;
    page_start: number | null;
    page_end: number | null;
    similarity_score: number | null;
  }>;
  grounded: boolean;
}

export interface ConversationCreateRequest {
  title?: string;
}

async function handleApiError(response: Response, fallbackMessage: string): Promise<never> {
  // Parse the unified error envelope: { detail, error: { code, message, request_id? } }.
  // Falls back gracefully for legacy shapes so no handler breaks.
  let structuredCode: string | undefined;
  let requestId: string | undefined;
  try {
    const errorData = await response.json();
    if (errorData && typeof errorData === 'object') {
      if (errorData.error && typeof errorData.error === 'object') {
        structuredCode = errorData.error.code;
        requestId = errorData.error.request_id;
      }
      if (typeof errorData.detail === 'string') {
        throw new ApiError(errorData.detail, response.status, structuredCode, requestId);
      } else if (Array.isArray(errorData.detail)) {
        const msg = errorData.detail.map((d: { msg?: string }) => d.msg || 'Validation error').join(', ');
        throw new ApiError(msg, response.status, structuredCode, requestId);
      } else if (errorData.error?.message) {
        throw new ApiError(errorData.error.message, response.status, structuredCode, requestId);
      } else if (errorData.message) {
        throw new ApiError(errorData.message, response.status, structuredCode, requestId);
      }
    }
  } catch (e) {
    if (e instanceof ApiError) {
      throw e;
    }
    if (e instanceof Error && e.message !== fallbackMessage) {
      throw e;
    }
  }
  throw new ApiError(fallbackMessage, response.status, structuredCode, requestId);
}

export class ApiError extends Error {
  status: number;
  code?: string;
  requestId?: string;

  constructor(message: string, status: number, code?: string, requestId?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }

  get isAuthError(): boolean {
    return this.status === 401;
  }

  get isForbidden(): boolean {
    return this.status === 403;
  }

  get isNotFound(): boolean {
    return this.status === 404;
  }

  get isRateLimited(): boolean {
    return this.status === 429;
  }

  get isServerError(): boolean {
    return this.status >= 500;
  }
}

export const api = {
  async health(): Promise<HealthResponse> {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) {
      await handleApiError(response, 'Health check failed');
    }
    return response.json();
  },

  async root(): Promise<{ name: string; version: string; status: string }> {
    const response = await fetch(`${API_URL}/`);
    if (!response.ok) {
      await handleApiError(response, 'API request failed');
    }
    return response.json();
  },

  // Authentication endpoints
  async register(data: RegisterData): Promise<UserResponse> {
    const response = await fetch(`${API_URL}/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
      credentials: 'include', // Include cookies
    });

    if (!response.ok) {
      await handleApiError(response, 'Registration failed');
    }

    return response.json();
  },

  async login(data: LoginData): Promise<UserResponse> {
    const response = await fetch(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
      credentials: 'include', // Include cookies
    });

    if (!response.ok) {
      await handleApiError(response, 'Login failed');
    }

    return response.json();
  },

  async logout(): Promise<void> {
    const response = await fetch(`${API_URL}/auth/logout`, {
      method: 'POST',
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Logout failed');
    }
  },

  async getCurrentUser(): Promise<UserResponse> {
    const response = await fetch(`${API_URL}/auth/me`, {
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Not authenticated');
    }

    return response.json();
  },

  async checkProtected(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/protected`, {
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Not authenticated');
    }

    return response.json();
  },

  // Document management endpoints
  async uploadDocument(file: File): Promise<DocumentResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/documents`, {
      method: 'POST',
      body: formData,
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Document upload failed');
    }

    return response.json();
  },

  async listDocuments(limit = 50, offset = 0, search?: string, statusFilter?: string): Promise<DocumentListResponse> {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (search && search.trim()) {
      params.set('search', search.trim());
    }
    if (statusFilter && statusFilter.trim()) {
      params.set('status', statusFilter.trim());
    }
    const response = await fetch(`${API_URL}/documents?${params.toString()}`, {
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch documents');
    }

    return response.json();
  },

  async getDocument(id: number): Promise<DocumentResponse> {
    const response = await fetch(`${API_URL}/documents/${id}`, {
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, `Document #${id} not found`);
    }

    return response.json();
  },

  async getDocumentBlob(id: number): Promise<Blob> {
    const response = await fetch(`${API_URL}/documents/${id}/file`, {
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Failed to download document file');
    }

    return response.blob();
  },

  getDocumentFileUrl(id: number): string {
    return `${API_URL}/documents/${id}/file`;
  },

  async deleteDocument(id: number): Promise<DeleteResponse> {
    const response = await fetch(`${API_URL}/documents/${id}`, {
      method: 'DELETE',
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Failed to delete document');
    }

    return response.json();
  },

  // Processing endpoints
  async getDocumentStatus(id: number): Promise<DocumentStatusResponse> {
    const response = await fetch(`${API_URL}/documents/${id}/status`, {
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, `Failed to get status for document #${id}`);
    }

    return response.json();
  },

  async getDocumentContent(id: number): Promise<DocumentContentResponse> {
    const response = await fetch(`${API_URL}/documents/${id}/content`, {
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, `Failed to get content for document #${id}`);
    }

    return response.json();
  },

  async retryProcessing(id: number): Promise<DocumentStatusResponse> {
    const response = await fetch(`${API_URL}/documents/${id}/process`, {
      method: 'POST',
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, `Failed to retry processing for document #${id}`);
    }

    return response.json();
  },

  // RAG endpoint
  async askQuestion(data: RAGRequest): Promise<RAGResponse> {
    const response = await fetch(`${API_URL}/rag/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Failed to process your question');
    }

    return response.json();
  },

  // Conversation endpoints
  async listConversations(limit = 20, offset = 0, search?: string): Promise<ConversationListResponse> {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (search && search.trim()) {
      params.set('search', search.trim());
    }
    const response = await fetch(`${API_URL}/conversations?${params.toString()}`, {
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch conversations');
    }

    return response.json();
  },

  async createConversation(data: ConversationCreateRequest = {}): Promise<ConversationResponse> {
    const response = await fetch(`${API_URL}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Failed to create conversation');
    }

    return response.json();
  },

  async getConversation(id: number): Promise<ConversationDetailResponse> {
    const response = await fetch(`${API_URL}/conversations/${id}`, {
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Conversation not found');
    }

    return response.json();
  },

  async getConversationMessages(
    conversationId: number,
    page = 1,
    pageSize = 50,
  ): Promise<PaginatedMessageResponse> {
    const response = await fetch(
      `${API_URL}/conversations/${conversationId}/messages?page=${page}&page_size=${pageSize}`,
      { credentials: 'include' },
    );

    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch messages');
    }

    return response.json();
  },

  async searchMessages(
    conversationId: number,
    query: string,
    page = 1,
    pageSize = 20,
  ): Promise<PaginatedMessageResponse> {
    const params = new URLSearchParams({
      q: query,
      page: String(page),
      page_size: String(pageSize),
    });
    const response = await fetch(
      `${API_URL}/conversations/${conversationId}/messages/search?${params.toString()}`,
      { credentials: 'include' },
    );

    if (!response.ok) {
      await handleApiError(response, 'Failed to search messages');
    }

    return response.json();
  },

  async sendMessage(
    conversationId: number,
    data: SendMessageRequest,
  ): Promise<SendMessageResponse> {
    const response = await fetch(`${API_URL}/conversations/${conversationId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Failed to send message');
    }

    return response.json();
  },

  async sendMessageStream(
    conversationId: number,
    data: SendMessageRequest,
    onToken: (text: string) => void,
    onSources: (sources: Array<{ document_id: number; filename: string | null; chunk_id: number; chunk_index: number; page_start: number | null; page_end: number | null; similarity_score: number | null; }>, confidence: { level: string; grounding_score: number; }) => void,
    onComplete: (messageId: number, grounded: boolean) => void,
    onError: (message: string) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const response = await fetch(`${API_URL}/conversations/${conversationId}/messages/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
      credentials: 'include',
      signal,
    });

    if (!response.ok) {
      await handleApiError(response, 'Failed to stream message');
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('ReadableStream not supported');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let eventType = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            try {
              const parsed = JSON.parse(dataStr);
              switch (eventType) {
                case 'token':
                  onToken(parsed.text || '');
                  break;
                case 'sources':
                  onSources(parsed.sources || [], parsed.confidence || { level: 'LOW', grounding_score: 0 });
                  break;
                case 'complete':
                  onComplete(parsed.message_id, parsed.grounded || false);
                  break;
                case 'error':
                  onError(parsed.message || 'An error occurred');
                  break;
              }
            } catch {
              // Skip malformed JSON
            }
            eventType = '';
          }
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        // User cancelled — not an error
        return;
      }
      throw err;
    }
  },

  async updateConversation(id: number, title: string): Promise<ConversationResponse> {
    const response = await fetch(`${API_URL}/conversations/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Failed to update conversation');
    }

    return response.json();
  },

  async deleteConversation(id: number): Promise<DeleteResponse> {
    const response = await fetch(`${API_URL}/conversations/${id}`, {
      method: 'DELETE',
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Failed to delete conversation');
    }

    return response.json();
  },  async exportConversation(id: number): Promise<void> {
    const response = await fetch(`${API_URL}/conversations/${id}/export`, {
      credentials: 'include',
    });

    if (!response.ok) {
      await handleApiError(response, 'Failed to export conversation');
    }

    const blob = await response.blob();
    const disposition = response.headers.get('Content-Disposition');
    let filename = 'conversation.md';
    if (disposition) {
      // Try RFC 5987 filename* first (properly encoded Unicode)
      const encodedMatch = disposition.match(/filename\*=UTF-8''(.+?)(?:;|$)/);
      if (encodedMatch) {
        filename = decodeURIComponent(encodedMatch[1].trim());
      } else {
        // Fall back to basic filename parameter
        const match = disposition.match(/filename="?([^";]+)"?(?:;|$)/);
        if (match) filename = match[1].trim();
      }
    }

    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  },

  // Collection endpoints
  async listCollections(limit = 50, offset = 0): Promise<{ items: Array<{ id: number; name: string; description: string | null; created_at: string; updated_at: string; document_count: number }>; total: number; limit: number; offset: number }> {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    const response = await fetch(`${API_URL}/collections?${params.toString()}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch collections');
    }
    return response.json();
  },

  async createCollection(name: string, description?: string): Promise<{ id: number; name: string; description: string | null; created_at: string; updated_at: string }> {
    const response = await fetch(`${API_URL}/collections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to create collection');
    }
    return response.json();
  },

  async deleteCollection(id: number): Promise<DeleteResponse> {
    const response = await fetch(`${API_URL}/collections/${id}`, {
      method: 'DELETE',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to delete collection');
    }
    return response.json();
  },

  async addDocumentToCollection(collectionId: number, documentId: number): Promise<DeleteResponse> {
    const response = await fetch(`${API_URL}/collections/${collectionId}/documents/${documentId}`, {
      method: 'POST',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to add document to collection');
    }
    return response.json();
  },

  async removeDocumentFromCollection(collectionId: number, documentId: number): Promise<DeleteResponse> {
    const response = await fetch(`${API_URL}/collections/${collectionId}/documents/${documentId}`, {
      method: 'DELETE',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to remove document from collection');
    }
    return response.json();
  },

  async getDocumentJobs(documentId: number): Promise<{ document_id: number; jobs: Array<{ id: number; status: string; attempts: number; max_attempts: number; error_message: string | null; created_at: string | null; started_at: string | null; completed_at: string | null }> }> {
    const response = await fetch(`${API_URL}/documents/${documentId}/jobs`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch document jobs');
    }
    return response.json();
  },

  // --- Phase 13: Enterprise platform endpoints ---

  async listWorkspaces(): Promise<{ items: Array<{ id: number; name: string; description: string; role?: string }> }> {
    const response = await fetch(`${API_URL}/workspaces`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch workspaces');
    }
    return response.json();
  },

  async createWorkspace(name: string, description?: string): Promise<{ id: number; name: string }> {
    const response = await fetch(`${API_URL}/workspaces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to create workspace');
    }
    return response.json();
  },

  // API keys
  async listApiKeys(workspaceId: number): Promise<{ items: Array<{
    id: number; name: string; prefix: string; workspace_id: number; scopes: string[];
    last_used_at: string | null; expires_at: string | null; revoked_at: string | null; created_at: string;
  }> }> {
    const response = await fetch(`${API_URL}/api-keys?workspace_id=${workspaceId}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch API keys');
    }
    return response.json();
  },

  async createApiKey(workspaceId: number, name: string, scopes: string[]): Promise<{
    id: number; name: string; prefix: string; key?: string; scopes: string[];
  }> {
    const response = await fetch(`${API_URL}/api-keys`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: workspaceId, name, scopes }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to create API key');
    }
    return response.json();
  },

  async revokeApiKey(id: number): Promise<{ message: string }> {
    const response = await fetch(`${API_URL}/api-keys/${id}/revoke`, {
      method: 'POST',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to revoke API key');
    }
    return response.json();
  },

  async listApiKeyScopes(): Promise<{ scopes: string[] }> {
    const response = await fetch(`${API_URL}/api-keys/scopes`, { credentials: 'include' });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch API key scopes');
    }
    return response.json();
  },

  // Webhooks
  async listWebhooks(workspaceId: number): Promise<{ items: Array<{
    id: number; url: string; events: string[]; status: string; description: string | null;
    workspace_id: number; created_at: string;
  }> }> {
    const response = await fetch(`${API_URL}/webhooks?workspace_id=${workspaceId}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch webhooks');
    }
    return response.json();
  },

  async createWebhook(workspaceId: number, url: string, events: string[]): Promise<{
    id: number; url: string; events: string[]; status: string; signing_secret?: string;
  }> {
    const response = await fetch(`${API_URL}/webhooks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: workspaceId, url, events }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to create webhook');
    }
    return response.json();
  },

  async deleteWebhook(id: number): Promise<{ message: string }> {
    const response = await fetch(`${API_URL}/webhooks/${id}`, {
      method: 'DELETE',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to delete webhook');
    }
    return response.json();
  },

  async listWebhookEvents(): Promise<{ events: string[] }> {
    const response = await fetch(`${API_URL}/webhooks/events`, { credentials: 'include' });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch webhook events');
    }
    return response.json();
  },

  async rotateWebhookSecret(id: number): Promise<{ message: string; signing_secret: string }> {
    const response = await fetch(`${API_URL}/webhooks/${id}/rotate-secret`, {
      method: 'POST',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to rotate webhook secret');
    }
    return response.json();
  },

  // Usage
  async getWorkspaceUsage(workspaceId: number): Promise<{
    usage: Record<string, number>;
    quota: Record<string, { used: number; limit: number | null; percent: number; remaining: number | null }>;
  }> {
    const response = await fetch(`${API_URL}/usage/workspaces/${workspaceId}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch usage');
    }
    return response.json();
  },

  async getWorkspaceLimits(workspaceId: number): Promise<{
    limits: Record<string, number>;
    features: Record<string, boolean>;
    plan: string;
  }> {
    const response = await fetch(`${API_URL}/usage/workspaces/${workspaceId}/limits`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch plan limits');
    }
    return response.json();
  },

  // Organizations
  async listOrganizations(): Promise<{ items: Array<{
    id: number; name: string; slug: string; status: string; role?: string; created_at: string;
  }> }> {
    const response = await fetch(`${API_URL}/organizations`, { credentials: 'include' });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch organizations');
    }
    return response.json();
  },

  async createOrganization(name: string, slug?: string): Promise<{
    id: number; name: string; slug: string; status: string;
  }> {
    const response = await fetch(`${API_URL}/organizations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, slug }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to create organization');
    }
    return response.json();
  },

  // Integrations
  async listIntegrationProviders(): Promise<{ items: Array<{
    provider: string; name: string; status: string; description: string; capabilities: string[];
  }> }> {
    const response = await fetch(`${API_URL}/integrations/providers`, { credentials: 'include' });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch integrations');
    }
    return response.json();
  },

  // --- Phase 14: AI product endpoints ---

  async getWorkspaceKnowledgeHealth(workspaceId: number): Promise<{
    total_documents: number;
    active_knowledge: number;
    stale_knowledge: number;
    processing_failures: number;
    metadata_completeness: number;
    duplicate_rate: number;
    conflict_rate: number;
    avg_health: number | null;
    ai_usage: { ai_executions_30d: number; ai_cost_30d: number; success_rate: number | null };
  }> {
    const response = await fetch(`${API_URL}/knowledge/workspaces/${workspaceId}/health`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch knowledge health');
    }
    return response.json();
  },

  async getActionCenterSummary(workspaceId: number): Promise<{
    status_counts: Record<string, number>;
    pending_actions: number;
    running: number;
    completed: number;
    failed: number;
  }> {
    const response = await fetch(`${API_URL}/ai-actions/summary?workspace_id=${workspaceId}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch action center');
    }
    return response.json();
  },

  async listAiActions(workspaceId: number, status?: string): Promise<{ items: Array<{
    id: number; action_type: string; title: string; status: string; risk_level: string;
    source_type: string; created_at: string;
  }> }> {
    const params = new URLSearchParams({ workspace_id: String(workspaceId) });
    if (status) params.set('status', status);
    const response = await fetch(`${API_URL}/ai-actions?${params.toString()}`, { credentials: 'include' });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch AI actions');
    }
    return response.json();
  },

  async listSuggestions(workspaceId: number): Promise<{ items: Array<{
    id: number; title: string; description: string | null; reason: string | null;
    suggestion_type: string; priority: string; status: string; created_at: string;
  }> }> {
    const response = await fetch(`${API_URL}/ai-actions/suggestions?workspace_id=${workspaceId}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch suggestions');
    }
    return response.json();
  },

  async generateSuggestions(workspaceId: number): Promise<{ suggestions_created: number }> {
    const response = await fetch(`${API_URL}/ai-actions/suggestions/generate?workspace_id=${workspaceId}`, {
      method: 'POST',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to generate suggestions');
    }
    return response.json();
  },

  async dismissSuggestion(suggestionId: number): Promise<{ message: string }> {
    const response = await fetch(`${API_URL}/ai-actions/suggestions/${suggestionId}/dismiss`, {
      method: 'POST',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to dismiss suggestion');
    }
    return response.json();
  },

  async askCopilot(question: string, scope = 'WORKSPACE'): Promise<{
    scope: string; command: string; answer: string; grounded: boolean;
    sources: Array<{ document_id: number; filename: string | null; similarity_score: number | null }>;
    confidence: { level: string; grounding_score: number; supporting_sources: number };
    matched_documents: number;
  }> {
    const response = await fetch(`${API_URL}/copilot/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, scope }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Copilot could not answer');
    }
    return response.json();
  },

  async analyzeSearchQuery(query: string): Promise<{
    query: string; intents: string[]; filters: Record<string, string>; explanations: string[];
  }> {
    const response = await fetch(`${API_URL}/search/intent?query=${encodeURIComponent(query)}`, {
      method: 'POST',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to analyze search query');
    }
    return response.json();
  },

  async listDeadlines(workspaceId: number, status?: string): Promise<{ items: Array<{
    id: number; title: string; due_date: string; status: string; confidence: string; source: string;
  }> }> {
    const params = new URLSearchParams({ workspace_id: String(workspaceId) });
    if (status) params.set('status', status);
    const response = await fetch(`${API_URL}/deadlines?${params.toString()}`, { credentials: 'include' });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch deadlines');
    }
    return response.json();
  },

  async createDeadline(workspaceId: number, title: string, dueDate: string): Promise<{
    id: number; title: string; due_date: string; status: string;
  }> {
    const response = await fetch(`${API_URL}/deadlines`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: workspaceId, title, due_date: dueDate }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to create deadline');
    }
    return response.json();
  },

  async submitFeedback(rating: 'thumbs_up' | 'thumbs_down', category?: string, comment?: string): Promise<{
    id: number; rating: string; status: string;
  }> {
    const response = await fetch(`${API_URL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rating, category, comment }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to submit feedback');
    }
    return response.json();
  },

  async getFeedbackAnalytics(): Promise<{
    total_feedback: number; acceptance_rate: number | null; thumbs_up: number; thumbs_down: number;
    categories: Record<string, number>;
  }> {
    const response = await fetch(`${API_URL}/feedback/analytics`, { credentials: 'include' });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch feedback analytics');
    }
    return response.json();
  },

  async listReportTemplates(): Promise<{ templates: Array<{ template: string; name: string; sections: string[] }> }> {
    const response = await fetch(`${API_URL}/reports/templates`, { credentials: 'include' });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch report templates');
    }
    return response.json();
  },

  async generateReport(template: string): Promise<{
    title: string; template: string; executive_summary: string;
    findings: Array<{ type: string; detail: string; source: string }>;
    recommendations: Array<{ type: string; detail: string; source: string }>;
    generated_at: string;
  }> {
    const response = await fetch(`${API_URL}/reports/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ template }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to generate report');
    }
    return response.json();
  },

  async runResearch(question: string): Promise<{
    question: string; answer: string; evidence: unknown[]; conflicts: unknown[];
    uncertainties: string[]; grounded: boolean; sources: unknown[];
  }> {
    const response = await fetch(`${API_URL}/research`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Research failed');
    }
    return response.json();
  },

  // --- Phase 15: Knowledge OS endpoints ---

  async listExecutions(workspaceId: number, status?: string): Promise<{
    items: Array<{ id: string; execution_type: string; task_type: string; status: string;
      priority: string; model: string | null; started_at: string | null; completed_at: string | null;
      latency_ms: number | null; actual_cost: number; created_at: string }>;
  }> {
    const query = status ? `&status=${status}` : '';
    const response = await fetch(`${API_URL}/executions?workspace_id=${workspaceId}${query}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch AI executions');
    }
    return response.json();
  },

  async createExecution(workspaceId: number, executionType: string, taskType: string): Promise<{
    id: string; status: string; priority: string;
  }> {
    const response = await fetch(`${API_URL}/executions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: workspaceId, execution_type: executionType, task_type: taskType }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to create execution');
    }
    return response.json();
  },

  async getEventSummary(workspaceId: number): Promise<{
    PENDING: number; PROCESSED: number; FAILED: number; DEAD: number;
  }> {
    const response = await fetch(`${API_URL}/events/summary?workspace_id=${workspaceId}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch event outbox');
    }
    return response.json();
  },

  async getReviewSummary(workspaceId: number): Promise<{
    pending: number; overdue: number; by_status: Record<string, number>;
  }> {
    const response = await fetch(`${API_URL}/reviews/summary?workspace_id=${workspaceId}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch review queue');
    }
    return response.json();
  },

  async getMemorySummary(workspaceId: number): Promise<{
    total: number; by_scope_type: Array<{ scope: string; type: string; count: number }>;
  }> {
    const response = await fetch(`${API_URL}/memory/summary?workspace_id=${workspaceId}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch AI memory');
    }
    return response.json();
  },

  async getTimeline(workspaceId: number, limit = 15): Promise<{
    items: Array<{ event_type: string; occurred_at: string; severity: string; title: string }>;
    count: number;
  }> {
    const response = await fetch(`${API_URL}/knowledge-os/timeline?workspace_id=${workspaceId}&limit=${limit}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch knowledge timeline');
    }
    return response.json();
  },

  async getCostSummary(workspaceId: number, days = 30): Promise<{
    total_cost_usd: number; total_tokens: number; execution_count: number;
    by_model: Record<string, number>; by_feature: Record<string, number>;
  }> {
    const response = await fetch(`${API_URL}/costs/workspaces/${workspaceId}/summary?days=${days}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch AI costs');
    }
    return response.json();
  },

  async createSnapshot(workspaceId: number, name: string): Promise<{
    id: number; name: string; created_at: string;
  }> {
    const response = await fetch(`${API_URL}/knowledge-os/snapshots?workspace_id=${workspaceId}&name=${encodeURIComponent(name)}`, {
      method: 'POST',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to create knowledge snapshot');
    }
    return response.json();
  },

  async simulateWorkflow(definition: Record<string, unknown>): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/automation/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ definition }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Workflow simulation failed');
    }
    return response.json();
  },

  async getQueueMetrics(workspaceId: number): Promise<{
    by_queue: Array<{ queue_name: string; depth: number; running: number; waiting: number; retrying: number; dead: number }>;
    by_tenant: Array<{ workspace_id: number | null; active: number; queued: number }>;
  }> {
    const response = await fetch(`${API_URL}/ops/queue-metrics?workspace_id=${workspaceId}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch queue metrics');
    }
    return response.json();
  },

  async getVectorStatus(): Promise<{
    backend: string;
    pgvector_available: boolean;
    json_fallback: boolean;
    note?: string;
  }> {
    const response = await fetch(`${API_URL}/ops/vector-status`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch vector status');
    }
    return response.json();
  },

  async getProviderStatus(workspaceId: number): Promise<{
    items: Array<{
      id: number;
      provider: string;
      model: string;
      status: string;
      circuit_state: string;
      consecutive_failures: number;
      success_count: number;
      failure_count: number;
      avg_latency_ms: number | null;
      last_error: string | null;
      last_checked_at: string | null;
    }>;
  }> {
    const response = await fetch(`${API_URL}/ops/provider-status?workspace_id=${workspaceId}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch provider status');
    }
    return response.json();
  },

  async getGatewayStatus(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops/gateway-status?workspace_id=${workspaceId}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch gateway status');
    }
    return response.json();
  },

  // --- Phase 17: distributed worker platform -------------------------------

  async getWorkerFleet(): Promise<{
    items: Array<{
      worker_id: string;
      status: string;
      hostname: string | null;
      pid: number | null;
      version: string | null;
      queue_name: string | null;
      current_job_type: string | null;
      current_job_id: number | null;
      started_at: string | null;
      last_heartbeat: string | null;
      stopped_at: string | null;
    }>;
    total: number;
  }> {
    const response = await fetch(`${API_URL}/worker-fleet`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch worker fleet');
    }
    return response.json();
  },

  async getAutoscaleSignals(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/autoscale-signals`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch autoscale signals');
    }
    return response.json();
  },

  async recoverStaleWorkers(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/workers/recover-stale`, {
      method: 'POST',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to recover stale workers');
    }
    return response.json();
  },

  async listDeadLetters(workspaceId: number, limit = 50): Promise<{
    items: Array<{
      id: number;
      queue_name: string;
      job_type: string;
      workspace_id: number;
      attempt: number;
      max_attempts: number;
      error_message: string | null;
      completed_at: string | null;
    }>;
    total: number;
  }> {
    const response = await fetch(
      `${API_URL}/dead-letters?workspace_id=${workspaceId}&limit=${limit}`,
      { credentials: 'include' },
    );
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch dead letters');
    }
    return response.json();
  },

  async requeueDeadLetter(jobId: number): Promise<{ id: number; status: string }> {
    const response = await fetch(`${API_URL}/dead-letters/${jobId}/requeue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reset_attempts: true }),
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to requeue dead letter');
    }
    return response.json();
  },

  async abandonDeadLetter(jobId: number): Promise<{ id: number; status: string }> {
    const response = await fetch(`${API_URL}/dead-letters/${jobId}/abandon`, {
      method: 'POST',
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to abandon dead letter');
    }
    return response.json();
  },

  async getVectorBackfillPreview(): Promise<{
    documents_to_process: number;
    chunks_to_embed: number;
    dry_run: boolean;
    native_pgvector: boolean;
    vector_backend: string;
    model: string;
    dimensions: number;
  }> {
    const response = await fetch(`${API_URL}/vector-backfill/preview`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch vector backfill preview');
    }
    return response.json();
  },

  async getCostAnomalies(workspaceId?: number): Promise<Array<Record<string, unknown>>> {
    const query = workspaceId != null ? `?workspace_id=${workspaceId}` : '';
    const response = await fetch(`${API_URL}/cost/anomalies${query}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch cost anomalies');
    }
    return response.json();
  },

  async getCostForecast(workspaceId?: number): Promise<Record<string, unknown>> {
    const query = workspaceId != null ? `?workspace_id=${workspaceId}` : '';
    const response = await fetch(`${API_URL}/cost/forecast${query}`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch cost forecast');
    }
    return response.json();
  },

  // --- Phase 18: enterprise operations console ------------------------------

  async getWorkerHealth(): Promise<{
    status: string;
    workers: number;
    active: number;
    stale: number;
    timestamp: number;
  }> {
    const response = await fetch(`${API_URL}/worker-health`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch worker health');
    }
    return response.json();
  },

  async getBrokerHealth(): Promise<{
    broker: string;
    status: string;
    latency_ms: number | null;
    queue_depth: number;
    errors: string[];
  }> {
    const response = await fetch(`${API_URL}/broker/health`, {
      credentials: 'include',
    });
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch broker health');
    }
    return response.json();
  },

  async getVectorHealth(workspaceId: number): Promise<{
    pgvector: Record<string, unknown>;
    backend: string;
    active_model: {
      id: number;
      provider: string;
      model: string;
      dimensions: number;
      version: string;
    } | null;
    total_chunks: number;
    vector_coverage_pct: number;
    index_params: Record<string, unknown>;
  }> {
    const response = await fetch(
      `${API_URL}/vector/health?workspace_id=${workspaceId}`,
      { credentials: 'include' },
    );
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch vector health');
    }
    return response.json();
  },

  async getPoisonDocuments(workspaceId: number): Promise<{
    items: Array<{
      id: number;
      workspace_id: number;
      document_id: number | null;
      stage: string;
      failure_count: number;
      last_error: string | null;
      status: string;
    }>;
    total: number;
  }> {
    const response = await fetch(
      `${API_URL}/ingestion/poison?workspace_id=${workspaceId}`,
      { credentials: 'include' },
    );
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch poison documents');
    }
    return response.json();
  },

  async getConnectorHealth(workspaceId: number): Promise<{
    items: Array<{
      id: number;
      name: string;
      kind: string;
      enabled: boolean;
      scopes: Record<string, unknown> | null;
      last_sync_status: string | null;
      lag_s: number | null;
      total_items: number;
      last_error: string | null;
    }>;
  }> {
    const response = await fetch(
      `${API_URL}/connectors/health?workspace_id=${workspaceId}`,
      { credentials: 'include' },
    );
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch connector health');
    }
    return response.json();
  },

  async getSloStatus(workspaceId: number): Promise<{
    status: string;
    availability: number | null;
    error_rate: number | null;
    targets: Record<string, number>;
    snapshots: number;
  }> {
    const response = await fetch(
      `${API_URL}/observability/slo?workspace_id=${workspaceId}`,
      { credentials: 'include' },
    );
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch SLO status');
    }
    return response.json();
  },

  async getCostAttribution(workspaceId: number): Promise<{
    days: number;
    total_cost: number;
    total_tokens: number;
    execution_count: number;
    by_workspace: Array<{ key: string; cost: number }>;
    by_user: Array<{ key: string; cost: number }>;
    by_feature: Array<{ key: string; cost: number }>;
    by_model: Array<{ key: string; cost: number }>;
  }> {
    const response = await fetch(
      `${API_URL}/cost/attribution?workspace_id=${workspaceId}`,
      { credentials: 'include' },
    );
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch cost attribution');
    }
    return response.json();
  },

  async getCostProjection(workspaceId: number): Promise<{
    is_estimate: boolean;
    assumptions: string[];
    last_30d_cost: number;
    daily_rate_est: number;
    projected_30d_cost_est: number;
    confidence: string;
  }> {
    const response = await fetch(
      `${API_URL}/cost/projection?workspace_id=${workspaceId}`,
      { credentials: 'include' },
    );
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch cost projection');
    }
    return response.json();
  },

  async getSearchAnalytics(workspaceId: number): Promise<{
    searches: number;
    zero_result_rate: number | null;
    p95_latency_ms: number | null;
    avg_latency_ms: number | null;
    by_mode: Record<string, number>;
    top_queries_by_hash: Array<[string, number]>;
  }> {
    const response = await fetch(
      `${API_URL}/search/analytics?workspace_id=${workspaceId}`,
      { credentials: 'include' },
    );
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch search analytics');
    }
    return response.json();
  },

  async getBackupInventory(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(
      `${API_URL}/dr/backup-inventory?workspace_id=${workspaceId}`,
      { credentials: 'include' },
    );
    if (!response.ok) {
      await handleApiError(response, 'Failed to fetch backup inventory');
    }
    return response.json();
  },

  async validateRestore(workspaceId: number): Promise<{
    migration_in_sync: boolean;
    migration_current: string | null;
    tenant_data: Record<string, number>;
    authorization_orphans: number;
    restore_valid: boolean;
  }> {
    const response = await fetch(
      `${API_URL}/dr/validate-restore?workspace_id=${workspaceId}`,
      { credentials: 'include' },
    );
    if (!response.ok) {
      await handleApiError(response, 'Failed to validate restore');
    }
    return response.json();
  },

  // --- Phase 19: control plane / operations center 4.0 --------------------

  async getOps19Health(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops19/health`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch global health');
    return response.json();
  },

  async getControlConfig(scopeType = 'GLOBAL'): Promise<{
    active_version: number | null;
    snapshots: Array<{ id: number; version: number; active: boolean; reason: string | null; created_at: string }>;
  }> {
    const response = await fetch(`${API_URL}/ops19/config?scope_type=${scopeType}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch control config');
    return response.json();
  },

  async getRegions(): Promise<Array<{
    region_id: string;
    name: string | null;
    status: string;
    health_score: number;
    failover_to: string | null;
  }>> {
    const response = await fetch(`${API_URL}/ops19/regions`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch regions');
    return response.json();
  },

  async getQuarantines(): Promise<Array<{
    worker_id: string;
    status: string;
    reason: string | null;
    failure_count: number;
    auto_recover_after: string | null;
  }>> {
    const response = await fetch(`${API_URL}/ops19/quarantines`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch quarantines');
    return response.json();
  },

  async getBroker2Info(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops19/broker`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch broker info');
    return response.json();
  },

  async getSchedulerLeader(): Promise<{ leader: { leader_id: string; lease_until: string } | null }> {
    const response = await fetch(`${API_URL}/ops19/scheduler/leader`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch scheduler leader');
    return response.json();
  },

  async getProviderCapabilities(): Promise<{
    matrix: Record<string, unknown>;
    registered: Array<Record<string, unknown>>;
  }> {
    const response = await fetch(`${API_URL}/ops19/providers/capabilities`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch provider capabilities');
    return response.json();
  },

  async getVectorCoverage(workspaceId?: number): Promise<Record<string, unknown>> {
    const query = workspaceId ? `?workspace_id=${workspaceId}` : '';
    const response = await fetch(`${API_URL}/ops19/vector/coverage${query}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch vector coverage');
    return response.json();
  },

  async getVectorDrift(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops19/vector/drift`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch vector drift');
    return response.json();
  },

  async getIngestionGovernor(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops19/ingestion/governor`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch ingestion governor');
    return response.json();
  },

  async getSloHealth(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops19/slo/health`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch SLO health');
    return response.json();
  },

  async runConsistencyCheck(workspaceId: number, dryRun = true): Promise<Record<string, unknown>> {
    const response = await fetch(
      `${API_URL}/ops19/consistency/run?workspace_id=${workspaceId}&dry_run=${dryRun}`,
      { method: 'POST', credentials: 'include' },
    );
    if (!response.ok) await handleApiError(response, 'Failed to run consistency check');
    return response.json();
  },

  async getDrReadiness(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops19/dr/readiness`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch DR readiness');
    return response.json();
  },

  async getSearchQuality(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops19/search/quality?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch search quality');
    return response.json();
  },

  async getQualityEvaluations(): Promise<Array<{
    id: number;
    dataset: string | null;
    passed: boolean;
    created_at: string;
  }>> {
    const response = await fetch(`${API_URL}/ops19/quality/evaluations`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch evaluations');
    return response.json();
  },

  // ---- Phase 20: self-improvement control center (ops20) ----

  async getImprovements(): Promise<Array<Record<string, unknown>>> {
    const response = await fetch(`${API_URL}/ops20/improvements`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch improvement proposals');
    return response.json();
  },

  async getExperiments(): Promise<Array<Record<string, unknown>>> {
    const response = await fetch(`${API_URL}/ops20/experiments`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch experiments');
    return response.json();
  },

  async getIncidents(): Promise<Array<Record<string, unknown>>> {
    const response = await fetch(`${API_URL}/ops20/incidents`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch incidents');
    return response.json();
  },

  async getAlerts(): Promise<Array<Record<string, unknown>>> {
    const response = await fetch(`${API_URL}/ops20/alerts`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch alerts');
    return response.json();
  },

  async getAlertSummary(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/alerts/summary`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch alert summary');
    return response.json();
  },

  async getSloHistory(): Promise<Array<Record<string, unknown>>> {
    const response = await fetch(`${API_URL}/ops20/slo/history`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch SLO history');
    return response.json();
  },

  async getQualityScorecards(workspaceId?: number): Promise<Array<Record<string, unknown>>> {
    const query = workspaceId ? `?workspace_id=${workspaceId}` : '';
    const response = await fetch(`${API_URL}/ops20/quality/scorecards${query}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch quality scorecards');
    return response.json();
  },

  async getKnowledgeHealth(workspaceId: number): Promise<Record<string, unknown> | null> {
    const response = await fetch(`${API_URL}/ops20/knowledge/health?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch knowledge health');
    return response.json();
  },

  async getKnowledgeGaps(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/knowledge/gaps?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch knowledge gaps');
    return response.json();
  },

  async getGovernanceAudit(): Promise<Array<Record<string, unknown>>> {
    const response = await fetch(`${API_URL}/ops20/governance/audit?scope_type=ORGANIZATION`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch governance audit');
    return response.json();
  },

  async getInjectionCorpus(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/safety/injection-corpus`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch safety corpus');
    return response.json();
  },

  async getApiHealth(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/api/health`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch API health');
    return response.json();
  },

  async getDatabaseHealth(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/database/health`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch database health');
    return response.json();
  },

  async getSystemHealth(factors: Record<string, number>): Promise<Record<string, unknown>> {
    const encoded = encodeURIComponent(JSON.stringify(factors));
    const response = await fetch(`${API_URL}/ops20/health/score?factors=${encoded}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch system health');
    return response.json();
  },

  async getDependencyGraph(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/dependency-graph`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch dependency graph');
    return response.json();
  },

  async getFeedbackSummary(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/feedback/summary?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch feedback summary');
    return response.json();
  },

  async getAgentIntelligence(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/agents/intelligence?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch agent intelligence');
    return response.json();
  },

  async getWorkflowIntelligence(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/workflows/intelligence?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch workflow intelligence');
    return response.json();
  },

  async getMemoryQuality(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/memory/quality?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch memory quality');
    return response.json();
  },

  async getGraphHealth(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/graph/health?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch graph health');
    return response.json();
  },

  async getSearchIntelligence(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops20/search/intelligence?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch search intelligence');
    return response.json();
  },

  async getReports(): Promise<Array<Record<string, unknown>>> {
    const response = await fetch(`${API_URL}/ops20/reports`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch reports');
    return response.json();
  },

  // ---- Phase 21: autonomous operations control center (ops21) ----

  async getAutonomyPolicies(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/autonomy/policies?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch autonomy policies');
    return response.json();
  },

  async getAutonomyOperations(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/autonomy/operations?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch autonomous operations');
    return response.json();
  },

  async simulateAutonomy(workspaceId: number, operationType: string): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/autonomy/simulate`, {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: workspaceId, operation_type: operationType }),
    });
    if (!response.ok) await handleApiError(response, 'Failed to simulate operation');
    return response.json();
  },

  async setAutonomyLevel(workspaceId: number, policyId: number, newLevel: string): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/autonomy/level`, {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: workspaceId, policy_id: policyId, new_level: newLevel }),
    });
    if (!response.ok) await handleApiError(response, 'Failed to set autonomy level');
    return response.json();
  },

  async getSystemHealthLatest(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/health/latest?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch system health');
    return response.json();
  },

  async getRecoveryAttempts(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/recovery/attempts?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch recovery attempts');
    return response.json();
  },

  async getDiagnoses(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/diagnosis?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch diagnoses');
    return response.json();
  },

  async getKnowledgeRecoveryPlans(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/knowledge/recovery-plans?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch knowledge recovery plans');
    return response.json();
  },

  async getAdaptiveCandidates(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/candidates?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch adaptive candidates');
    return response.json();
  },

  async getIncidentsP21(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/incidents?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch incidents');
    return response.json();
  },

  async emergencyStop(workspaceId: number, scope: string, reason: string): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/safety/emergency-stop`, {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: workspaceId, scope, reason }),
    });
    if (!response.ok) await handleApiError(response, 'Failed to activate emergency stop');
    return response.json();
  },

  async getActivityFeed(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops21/personal/activity?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch activity feed');
    return response.json();
  },

  // ---- Phase 22: production cloud + continuous operations (ops22) ----

  async getInfrastructure(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/infrastructure`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch infrastructure capabilities');
    return response.json();
  },

  async getVectorBackend(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/vector/backend`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch vector backend');
    return response.json();
  },

  async getVectorDriftP22(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/vector/drift/${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch vector drift');
    return response.json();
  },

  async benchmarkVector(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/vector/benchmark/${workspaceId}`, { method: 'POST', credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to run vector benchmark');
    return response.json();
  },

  async getProviderHealthP22(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/providers/health?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch provider health');
    return response.json();
  },

  async runEvaluation(workspaceId: number, datasetId: number, config: string): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/evaluations/run`, {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: workspaceId, dataset_id: datasetId, config }),
    });
    if (!response.ok) await handleApiError(response, 'Failed to run evaluation');
    return response.json();
  },

  async getEvaluationRegressions(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/evaluations/regression/${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch evaluation regressions');
    return response.json();
  },

  async getRegionsP22(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/regions?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch regions');
    return response.json();
  },

  async getDrReport(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/dr/report?workspace_id=${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch DR report');
    return response.json();
  },

  async runChaos(workspaceId: number, scenario: string): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/chaos/run`, {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: workspaceId, scenarios: [scenario] }),
    });
    if (!response.ok) await handleApiError(response, 'Failed to run chaos probe');
    return response.json();
  },

  async runLoad(workspaceId: number, scenario: string, volume: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/load/run`, {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: workspaceId, scenario, volume }),
    });
    if (!response.ok) await handleApiError(response, 'Failed to run load test');
    return response.json();
  },

  async getOpsStream(workspaceId: number, stream: string): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops22/streams/${stream}?workspace_id=${workspaceId}&limit=25`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch ops stream');
    return response.json();
  },

  // ---- Phase 23: global AI cloud platform (ops23) ----
  async getCapabilities(): Promise<Record<string, unknown>[]> {
    const response = await fetch(`${API_URL}/ops23/capabilities`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch capabilities');
    return response.json();
  },

  async getGlobalHealth(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops23/global-health`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch global health');
    return response.json();
  },

  async getDependencyImpact(component: string): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops23/dependencies/${component}/impact`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch dependency impact');
    return response.json();
  },

  async getProviderReadiness(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops23/providers/readiness`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch provider readiness');
    return response.json();
  },

  async getReconciliation(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops23/cost/reconcile/${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch reconciliation');
    return response.json();
  },

  async getResidencyLog(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops23/residency/log/${workspaceId}`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch residency log');
    return response.json();
  },

  async runSecurityScan(workspaceId: number): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops23/security/scan/${workspaceId}`, {
      method: 'POST', credentials: 'include',
    });
    if (!response.ok) await handleApiError(response, 'Failed to run security scan');
    return response.json();
  },

  async getVectorActivation(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_URL}/ops23/vector/activation`, { credentials: 'include' });
    if (!response.ok) await handleApiError(response, 'Failed to fetch vector activation');
    return response.json();
  },
};

export default api;

