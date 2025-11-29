from fastapi import FastAPI
from api.routers import users, topics, files, flashcards
import os
from pathlib import Path


app = FastAPI(title="Flashcard API")

# Create upload directory (relative to api folder)
API_DIR = Path(__file__).parent
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(API_DIR / "uploads"))
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# App routers
app.include_router(users.router)
app.include_router(topics.router)
app.include_router(files.router)
app.include_router(flashcards.router)

@app.get("/")
def root():
    return {"message": "Flashcard API"}
