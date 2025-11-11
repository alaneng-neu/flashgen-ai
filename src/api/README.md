# Flashcard API

A FastAPI application for managing flashcards, topics, and file uploads with AI-powered flashcard generation.

## Setup

### 1. Start PostgreSQL with Docker

```bash
docker run --name flashcard-db -e POSTGRES_USER=flashuser -e POSTGRES_PASSWORD=flashpass -e POSTGRES_DB=flashcard_db -p 5432:5432 -d postgres
```

**Verify the database is running:**

```bash
docker ps
```

### 2. Install Dependencies (at project root)

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the API root:

```env
DATABASE_URL=postgresql://flashuser:flashpass@localhost:5432/flashcard_db
UPLOAD_DIR=./uploads
```

### 4. Database Migration

Initialize Alembic (first time only):

```bash
alembic init migrations
```

Update `alembic.ini`:

```ini
sqlalchemy.url = postgresql://flashuser:flashpass@localhost:5432/flashcarddb
```

Update `migrations/env.py` to import your models and sqlmodel:

```python
import sqlmodel  # Add this import at the top

# ... other imports ...

from models import SQLModel
target_metadata = SQLModel.metadata
```

Create and run migration:

```bash
# Generate migration
alembic revision --autogenerate -m "Initial migration"
```

Before running the migration, open the generated migration file in `migrations/versions/` and add `import sqlmodel` at the top with the other imports:

```python
"""Initial migration

Revision ID: xxxxx
...
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel  # ADD THIS LINE

# ... rest of file
```

Now apply the migration:

```bash
alembic upgrade head
```

### 5. Run the API

```bash
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`

Interactive docs at `http://localhost:8000/docs`

---

## Development Workflow

### Typical Usage Flow

1. **Create a user**

   ```bash
   curl -X POST "http://localhost:8000/users" \
     -H "Content-Type: application/json" \
     -d '{"username": "student1"}'
   ```

2. **Create a topic for the user**

   ```bash
   curl -X POST "http://localhost:8000/users/1/topics" \
     -H "Content-Type: application/json" \
     -d '{"title": "Machine Learning Basics"}'
   ```

3. **Upload reference files (optional)**

   ```bash
   curl -X POST "http://localhost:8000/topics/{topic_id}/upload" \
     -F "user_id=1" \
     -F "files=@textbook.pdf"
   ```

4. **Generate flashcards from files**

   ```bash
   curl -X POST "http://localhost:8000/topics/{topic_id}/generate-flashcards" \
     -F "user_id=1" \
     -F "title=Week 1 Flashcards" \
     -F "files=@lecture_notes.pdf"
   ```

5. **Retrieve and study flashcards**
   ```bash
   curl -X GET "http://localhost:8000/flashcard-sets/{set_id}/flashcards"
   ```

---

## Stopping the Database

```bash
docker stop flashcard-db
docker rm flashcard-db
```

## Project Structure

```
.
├── main.py              # FastAPI app and routes
├── models.py            # SQLModel database models
├── schemas.py           # Pydantic schemas for API
├── database.py          # Database connection
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
├── uploads/             # Uploaded files directory
└── migrations/          # Alembic migrations
```
