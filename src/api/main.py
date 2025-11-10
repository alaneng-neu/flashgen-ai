from fastapi import FastAPI
from routers import users, topics, files, flashcards
import os
from pathlib import Path


app = FastAPI(title="Flashcard API")

# Create upload directory
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# App routers
app.include_router(users.router)
app.include_router(topics.router)
app.include_router(files.router)
app.include_router(flashcards.router)

@app.get("/")
def root():
    return {"message": "Flashcard API"}
