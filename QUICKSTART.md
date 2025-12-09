# Quickstart Guide

## Prerequisites

- **Python 3.10+**
- **Docker** - for running PostgreSQL database
- **Ollama** - for LLM-powered flashcard generation

## Installation

1. **Clone the repository and install dependencies:**

```bash
pip install -r requirements.txt
```

## Optional: Preprocessing Step

The repository includes a pre-built vector database, so this step is **not required** for basic usage.

If you want to rebuild the vector database from scratch, run the preprocessing notebook:

```bash
jupyter notebook src/preprocess_flashcards.ipynb
```

## Running the API

### 1. Start PostgreSQL Database

```bash
docker run --name flashcard-db -e POSTGRES_USER=flashuser -e POSTGRES_PASSWORD=flashpass -e POSTGRES_DB=flashcard_db -p 5432:5432 -d postgres
```

### 2. Apply Database Migration

From the `src/` directory:

```bash
cd src
alembic -c api/alembic.ini upgrade head
```

### 3. Start Ollama

Make sure Ollama is running with the required model. The API defaults to `qwen3:8b`.

To use a different model, edit `src/api/llm_client.py` and change the model name.

### 4. Start the API Server

From the `src/` directory:

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

## Using the Application

### Option 1: Interactive API Documentation (Recommended)

Access the auto-generated Swagger docs at:

```
http://localhost:8000/docs
```

### Option 2: Flashcard UI

Open `src/flashcard-ui.html` directly in your browser.

---

## Quick API Usage Example

1. **Create a user:**

   ```bash
   curl -X POST "http://localhost:8000/users" -H "Content-Type: application/json" -d '{"username": "student1"}'
   ```

2. **Create a topic:**

   ```bash
   curl -X POST "http://localhost:8000/users/1/topics" -H "Content-Type: application/json" -d '{"title": "AI Fundamentals"}'
   ```

3. **Upload reference files for the topic:**

   ```bash
   curl -X POST "http://localhost:8000/topics/1/upload" -F "user_id=1" -F "files=@textbook.pdf"
   ```

4. **Generate flashcards from a file:**
   ```bash
   curl -X POST "http://localhost:8000/topics/1/generate-flashcards" -F "user_id=1" -F "title=Week 1" -F "files=@notes.pdf"
   ```

For more detailed API documentation, see [src/api/README.md](src/api/README.md).

## Stopping the Database

```bash
docker stop flashcard-db
docker rm flashcard-db
```
