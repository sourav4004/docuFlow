# DocuFlow Architecture

## Overview

DocuFlow is an AI-powered document operations platform built with a modern, scalable architecture separating frontend, backend, and data layers.

## System Architecture

```
┌─────────────────┐
│   Next.js       │  Frontend (TypeScript)
│   (Port 3000)   │  - React UI
└────────┬────────┘  - Client-side state
         │           - API integration
         │ HTTP/REST
         │
┌────────▼────────┐
│   FastAPI       │  Backend (Python)
│   (Port 8000)   │  - Business logic
└────────┬────────┘  - API endpoints
         │           - Data validation
         │ SQLAlchemy
         │
┌────────▼────────┐
│  PostgreSQL     │  Database
│   (Port 5432)   │  - Relational data
└─────────────────┘  - Persistence
```

## Components

### Frontend (Next.js + TypeScript)

**Location:** `frontend/`

**Responsibilities:**
- User interface rendering
- Client-side interactivity
- API communication
- Environment-based configuration

**Technology Stack:**
- Next.js 15 with App Router
- TypeScript
- Tailwind CSS
- React Server Components

**Key Features:**
- Landing page with system overview
- Real-time backend health monitoring
- Environment variable configuration
- Dark mode support

**API Integration:**
The frontend communicates with the backend through a centralized API client (`lib/api.ts`) that uses the `NEXT_PUBLIC_API_URL` environment variable.

### Backend (FastAPI + Python)

**Location:** `backend/`

**Responsibilities:**
- RESTful API endpoints
- Business logic execution
- Database operations
- Request validation
- CORS configuration

**Technology Stack:**
- FastAPI
- SQLAlchemy (ORM)
- Alembic (migrations)
- Pydantic (validation)
- psycopg2 (PostgreSQL driver)

**Structure:**
```
backend/
├── app/
│   ├── main.py           # Application entry point
│   ├── api/              # API endpoints
│   │   ├── __init__.py
│   │   └── health.py     # Health check endpoint
│   ├── core/             # Core configuration
│   │   ├── config.py     # Settings management
│   │   └── database.py   # Database connection
│   ├── models/           # SQLAlchemy models (future)
│   ├── schemas/          # Pydantic schemas (future)
│   └── services/         # Business logic (future)
├── tests/                # Test suite
├── alembic/              # Database migrations
└── requirements.txt      # Python dependencies
```

**Current Endpoints:**
- `GET /` - API information
- `GET /health` - Health check with database status

### Database (PostgreSQL)

**Location:** Docker container (via docker-compose.yml)

**Responsibilities:**
- Data persistence
- Relational data storage
- Transaction management

**Configuration:**
- Image: postgres:15-alpine
- Port: 5432
- Database: docuflow_db
- Health checks enabled

**Connection:**
The backend connects using SQLAlchemy with connection pooling configured for production readiness.

## Environment Configuration

All components use environment variables for configuration, preventing hardcoded secrets and enabling environment-specific settings.

### Backend (.env)
```
DATABASE_URL=postgresql://user:pass@host:port/db
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Database (docker-compose.yml)
```
POSTGRES_USER=docuflow_user
POSTGRES_PASSWORD=docuflow_password
POSTGRES_DB=docuflow_db
```

## Local Development Setup

1. **Start PostgreSQL:**
   ```bash
   docker compose up -d
   ```

2. **Start Backend:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   uvicorn app.main:app --reload
   ```

3. **Start Frontend:**
   ```bash
   cd frontend
   npm install
   cp .env.example .env.local
   npm run dev
   ```

## API Communication Flow

```
User Browser → Next.js (localhost:3000)
    ↓
    HTTP GET /health
    ↓
FastAPI (localhost:8000) → PostgreSQL
    ↓
    SQL: SELECT 1
    ↓
Response: { "status": "ok", "database": "connected" }
```

## CORS Configuration

The backend is configured to accept requests from the frontend origin:
- Development: `http://localhost:3000`
- Configurable via `CORS_ORIGINS` environment variable
- Allows credentials for future authentication

## Database Migrations

Alembic is configured for managing database schema changes:

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Testing Strategy

### Backend Testing
- pytest for test framework
- TestClient for API testing
- Tests verify endpoints and response structure

### Frontend Testing
- ESLint for code quality
- TypeScript for type safety
- Build verification

## Future Components (Planned)

The following components are planned but **NOT YET IMPLEMENTED**:

### Phase 1: Authentication
- User registration and login
- JWT token management
- Session handling
- User model and database tables

### Phase 2: Document Management
- File upload (multipart)
- S3 storage integration
- Document metadata tracking
- Document retrieval

### Phase 3: AI Processing
- LLM integration (Claude/OpenAI)
- Document OCR
- Structured data extraction
- Processing queue (Redis + Celery)

### Phase 4: Advanced Features
- Search functionality
- Deadline tracking
- Email notifications (Gmail integration)
- Approval workflows
- External service integrations
- AI agent interface

## Security Considerations

**Current Implementation:**
- Environment variables for secrets
- CORS configured for specific origins
- Database connection pooling
- Input validation via Pydantic

**Future Requirements:**
- Authentication and authorization
- API rate limiting
- File upload validation
- Encryption at rest
- Audit logging

## Deployment Considerations (Future)

This foundation is prepared for:
- Container orchestration (Docker)
- Environment-specific configurations
- Database migration management
- Horizontal scaling
- Health monitoring

## Phase Status

**Current Phase:** Phase 0 - Project Foundation

**Completed:**
- ✅ Project structure
- ✅ Frontend (Next.js)
- ✅ Backend (FastAPI)
- ✅ Database (PostgreSQL)
- ✅ API communication
- ✅ Environment configuration
- ✅ CORS setup
- ✅ Health monitoring
- ✅ Testing framework

**Not Implemented:**
- ❌ Authentication
- ❌ User management
- ❌ Document upload
- ❌ AI processing
- ❌ Workflows
- ❌ Integrations
