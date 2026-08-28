# DocuFlow

AI-powered document operations platform for intelligent document management, processing, and workflow automation.

## Overview

DocuFlow is a modern, full-stack application that will enable users to:

- 📄 Upload and manage documents
- 🤖 Extract structured information using AI
- 🔍 Search and retrieve documents
- ⏰ Track deadlines and automate reminders
- ✅ Manage approval workflows
- 🔗 Integrate with external services

**Current Status:** Phase 0 - Project Foundation  
**Note:** Authentication, document management, and AI features are not yet implemented.

## Architecture

DocuFlow uses a three-tier architecture:

```
Next.js (Frontend) → FastAPI (Backend) → PostgreSQL (Database)
```

### Technology Stack

**Frontend:**
- Next.js 15 with App Router
- TypeScript
- Tailwind CSS
- React

**Backend:**
- FastAPI (Python)
- SQLAlchemy + Alembic
- Pydantic
- uvicorn

**Database:**
- PostgreSQL 15

**Infrastructure:**
- Docker Compose (local development)

## Requirements

- Node.js 18+ and npm
- Python 3.11+
- Docker and Docker Compose
- Git

## Project Structure

```
docuflow/
├── frontend/              # Next.js frontend application
│   ├── app/              # Next.js App Router pages
│   ├── lib/              # API client and utilities
│   ├── public/           # Static assets
│   └── package.json
│
├── backend/              # FastAPI backend application
│   ├── app/
│   │   ├── api/         # API endpoints
│   │   ├── core/        # Configuration and database
│   │   ├── models/      # SQLAlchemy models
│   │   ├── schemas/     # Pydantic schemas
│   │   └── services/    # Business logic
│   ├── tests/           # Backend tests
│   ├── alembic/         # Database migrations
│   └── requirements.txt
│
├── infrastructure/       # Infrastructure configuration
├── docs/                # Documentation
│   └── architecture.md  # Detailed architecture docs
│
├── docker-compose.yml   # Local PostgreSQL setup
├── .env.example         # Environment variables template
├── .gitignore
└── README.md
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project1
```

### 2. Configure Environment Variables

**Root Level:**
```bash
cp .env.example .env
# Edit .env if needed (defaults work for local development)
```

**Backend:**
```bash
cd backend
cp .env.example .env
cd ..
```

**Frontend:**
```bash
cd frontend
cp .env.example .env.local
cd ..
```

### 3. Start PostgreSQL

```bash
docker compose up -d
```

Verify PostgreSQL is running:
```bash
docker compose ps
```

### 4. Setup and Start Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at: http://localhost:8000

API documentation: http://localhost:8000/docs

### 5. Setup and Start Frontend

Open a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at: http://localhost:3000

## Testing

### Backend Tests

```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
pytest
```

### Frontend Linting and Build

```bash
cd frontend
npm run lint
npm run build
```

## API Endpoints

### Current Endpoints

**GET /**
```json
{
  "name": "DocuFlow API",
  "version": "0.1.0",
  "status": "running"
}
```

**GET /health**
```json
{
  "status": "ok",
  "database": "connected"
}
```

## Environment Variables

### Backend

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://docuflow_user:...@localhost:5432/docuflow_db` |
| `BACKEND_HOST` | Backend host address | `0.0.0.0` |
| `BACKEND_PORT` | Backend port | `8000` |
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `http://localhost:3000` |

### Frontend

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |

### Database (Docker)

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | Database user | `docuflow_user` |
| `POSTGRES_PASSWORD` | Database password | `docuflow_password` |
| `POSTGRES_DB` | Database name | `docuflow_db` |
| `POSTGRES_PORT` | Database port | `5432` |

## Database Migrations

Alembic is configured for database schema management.

**Create a new migration:**
```bash
cd backend
alembic revision --autogenerate -m "Description of changes"
```

**Apply migrations:**
```bash
alembic upgrade head
```

**Rollback migration:**
```bash
alembic downgrade -1
```

## Development Workflow

1. **Start the database:** `docker compose up -d`
2. **Start the backend:** Terminal 1 - `cd backend && uvicorn app.main:app --reload`
3. **Start the frontend:** Terminal 2 - `cd frontend && npm run dev`
4. **Make changes** and see them live-reload
5. **Run tests** before committing

## Current Phase: Phase 0 - Foundation

This is the foundational phase establishing the project structure and basic connectivity.

### ✅ Implemented

- Project structure and organization
- Next.js frontend with TypeScript
- FastAPI backend with Python
- PostgreSQL database via Docker
- Environment variable configuration
- CORS setup
- Health check endpoint
- Frontend ↔ Backend communication
- Basic testing framework
- Documentation

### ❌ Not Yet Implemented

The following features are planned for future phases:

- **Authentication:** User registration, login, JWT tokens
- **User Management:** User profiles, permissions
- **Document Upload:** File upload, storage (S3), metadata
- **Document Management:** List, retrieve, delete documents
- **AI Processing:** LLM integration, OCR, data extraction
- **Search:** Full-text search, filtering
- **Workflows:** Approval processes, automation
- **Notifications:** Email alerts, reminders
- **Integrations:** Gmail, Google Drive, external APIs
- **AI Agent:** Conversational document assistant

## Troubleshooting

### PostgreSQL Connection Issues

```bash
# Check if PostgreSQL is running
docker compose ps

# View PostgreSQL logs
docker compose logs postgres

# Restart PostgreSQL
docker compose restart postgres
```

### Backend Won't Start

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Windows: venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt

# Check for port conflicts
# Ensure port 8000 is not in use
```

### Frontend Won't Start

```bash
# Clear Next.js cache
rm -rf .next

# Reinstall dependencies
rm -rf node_modules
npm install

# Check for port conflicts
# Ensure port 3000 is not in use
```

### Frontend Can't Connect to Backend

1. Verify backend is running: http://localhost:8000/docs
2. Check `.env.local` has correct `NEXT_PUBLIC_API_URL`
3. Verify CORS settings in backend
4. Check browser console for errors

## Documentation

- **[Architecture Documentation](docs/architecture.md)** - Detailed architecture overview
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when backend is running)

## Contributing

This is a phased development project. Current phase focuses on foundation only.

## License

[To be determined]

## Contact

[To be determined]
