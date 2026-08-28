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

async function handleApiError(response: Response, fallbackMessage: string): Promise<never> {
  try {
    const errorData = await response.json();
    if (errorData && typeof errorData.detail === 'string') {
      throw new Error(errorData.detail);
    } else if (errorData && Array.isArray(errorData.detail)) {
      const msg = errorData.detail.map((d: { msg?: string }) => d.msg || 'Validation error').join(', ');
      throw new Error(msg);
    } else if (errorData && errorData.message) {
      throw new Error(errorData.message);
    }
  } catch (e) {
    if (e instanceof Error && e.message !== fallbackMessage) {
      throw e;
    }
  }
  throw new Error(fallbackMessage);
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

  async listDocuments(limit = 50, offset = 0): Promise<DocumentListResponse> {
    const response = await fetch(`${API_URL}/documents?limit=${limit}&offset=${offset}`, {
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
};

export default api;

