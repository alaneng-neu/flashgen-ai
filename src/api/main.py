from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from sqlmodel import Session, select
from database import get_session
from models import FlashcardTypeEnum, User, Topic, FlashcardSet, Flashcard, UploadedFile
from schemas import (
    UserCreate, UserRead,
    TopicCreate, TopicRead,
    FlashcardSetCreate, FlashcardSetRead,
    FlashcardRead,
    UploadedFileRead
)
from typing import List
from uuid import UUID
import os
import shutil
from pathlib import Path


app = FastAPI(title="Flashcard API")

# Create upload directory
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Flashcard API"}


# ==================== User Routes ====================

@app.post("/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, session: Session = Depends(get_session)):
    # Check if username exists
    existing_user = session.exec(select(User).where(User.username == user.username)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    db_user = User(username=user.username)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user

@app.get("/users/{user_id}", response_model=UserRead)
def get_user(user_id: int, session: Session = Depends(get_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ==================== Topic Routes ====================

@app.post("/users/{user_id}/topics", response_model=TopicRead, status_code=status.HTTP_201_CREATED)
def create_topic(user_id: int, topic: TopicCreate, session: Session = Depends(get_session)):
    # Verify user exists
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db_topic = Topic(title=topic.title, user_id=user_id)
    session.add(db_topic)
    session.commit()
    session.refresh(db_topic)
    return db_topic

@app.get("/topics/{topic_id}", response_model=TopicRead)
def get_topic(topic_id: UUID, session: Session = Depends(get_session)):
    topic = session.get(Topic, topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    return topic

@app.get("/users/{user_id}/topics", response_model=List[TopicRead])
def get_user_topics(user_id: int, session: Session = Depends(get_session)):
    statement = select(Topic).where(Topic.user_id == user_id)
    topics = session.exec(statement).all()
    return topics


# ==================== File Upload Route ====================

@app.post("/topics/{topic_id}/upload", response_model=List[UploadedFileRead], status_code=status.HTTP_201_CREATED)
async def upload_files(
    topic_id: UUID,
    user_id: int = Form(...),
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session)
):
    # Verify user exists
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Verify topic exists and belongs to user
    topic = session.get(Topic, topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    if topic.user_id != user_id:
        raise HTTPException(status_code=403, detail="Topic does not belong to user")
    
    # Create topic-specific directory
    topic_dir = Path(UPLOAD_DIR) / str(topic_id)
    topic_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    
    for file in files:
        # Generate unique filename to avoid collisions
        file_path = topic_dir / file.filename
        counter = 1
        while file_path.exists():
            name, ext = os.path.splitext(file.filename)
            file_path = topic_dir / f"{name}_{counter}{ext}"
            counter += 1
        
        # Save file to disk
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Save file record to database
        db_file = UploadedFile(
            file_name=file.filename,
            file_path=str(file_path),
            file_size=file_size,
            user_id=user_id,
            topic_id=topic_id
        )
        session.add(db_file)
        uploaded_files.append(db_file)
    
    # TODO: Generate system prompt based on uploaded files
    # Example: Analyze file contents, extract key concepts, generate prompt
    # topic.system_prompt = generate_system_prompt(uploaded_files)
    
    session.commit()
    
    # Refresh all uploaded files to get their IDs
    for db_file in uploaded_files:
        session.refresh(db_file)
    
    return uploaded_files

@app.get("/topics/{topic_id}/files", response_model=List[UploadedFileRead])
def get_topic_files(topic_id: UUID, session: Session = Depends(get_session)):
    statement = select(UploadedFile).where(UploadedFile.topic_id == topic_id)
    files = session.exec(statement).all()
    return files


# ==================== Flashcard Set Routes ====================

@app.post("/topics/{topic_id}/flashcard-sets", response_model=FlashcardSetRead, status_code=status.HTTP_201_CREATED)
def create_flashcard_set(topic_id: UUID, flashcard_set: FlashcardSetCreate, session: Session = Depends(get_session)):
    # Verify topic exists
    topic = session.get(Topic, topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    db_set = FlashcardSet(title=flashcard_set.title, topic_id=topic_id)
    session.add(db_set)
    session.commit()
    session.refresh(db_set)
    return db_set

@app.get("/flashcard-sets/{set_id}", response_model=FlashcardSetRead)
def get_flashcard_set(set_id: UUID, session: Session = Depends(get_session)):
    flashcard_set = session.get(FlashcardSet, set_id)
    if not flashcard_set:
        raise HTTPException(status_code=404, detail="Flashcard set not found")
    return flashcard_set

@app.get("/topics/{topic_id}/flashcard-sets", response_model=List[FlashcardSetRead])
def get_topic_flashcard_sets(topic_id: UUID, session: Session = Depends(get_session)):
    statement = select(FlashcardSet).where(FlashcardSet.topic_id == topic_id)
    sets = session.exec(statement).all()
    return sets


# ==================== Generate Flashcard Set ====================

@app.post("/topics/{topic_id}/generate-flashcards", response_model=List[FlashcardRead], status_code=status.HTTP_201_CREATED)
async def generate_flashcard_set(
    topic_id: UUID,
    user_id: int = Form(...),
    title: str = Form(...),
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session)
):
    # Verify user exists
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Verify topic exists and belongs to user
    topic = session.get(Topic, topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    if topic.user_id != user_id:
        raise HTTPException(status_code=403, detail="Topic does not belong to user")
    
    # Create the flashcard set
    db_set = FlashcardSet(title=title, topic_id=topic_id)
    session.add(db_set)
    session.commit()
    session.refresh(db_set)
    
    # Read file contents into memory
    file_contents = []
    for file in files:
        content = await file.read()
        file_contents.append({
            "filename": file.filename,
            "content_type": file.content_type,
            "content": content  # bytes
        })
        # Reset file pointer if you need to read again
        await file.seek(0)
    
    # TODO: Generate flashcards using:
    # - topic.system_prompt (context about the topic) -> topic.system_prompt
    # - file_contents (list of dicts with filename, content_type, and content bytes)
    # - AI/LLM integration to parse content and create flashcards
    #
    # Example pseudocode:
    # generated_flashcards = generate_with_llm(
    #     system_prompt=topic.system_prompt,
    #     files=file_contents,
    #     num_flashcards=10
    # )
    
    # Placeholder: Create sample flashcards for demonstration
    # Replace this with actual generation logic
    generated_flashcards = [
        {
            "term": "Sample Term 1",
            "definition": "Sample Definition 1",
            "flashcard_type": FlashcardTypeEnum.TERM_DEFINITION
        },
        {
            "term": "Sample Term 2",
            "definition": "Sample Definition 2",
            "flashcard_type": FlashcardTypeEnum.TERM_DEFINITION
        }
    ]
    
    # Save generated flashcards to database
    db_flashcards = []
    for fc_data in generated_flashcards:
        db_flashcard = Flashcard(
            term=fc_data["term"],
            definition=fc_data["definition"],
            flashcard_type=fc_data["flashcard_type"],
            flashcard_set_id=db_set.id
        )
        session.add(db_flashcard)
        db_flashcards.append(db_flashcard)
    
    session.commit()
    
    # Refresh all flashcards to get their IDs
    for db_flashcard in db_flashcards:
        session.refresh(db_flashcard)
    
    return db_flashcards


# ==================== Flashcard Routes ====================

@app.get("/flashcard-sets/{set_id}/flashcards", response_model=List[FlashcardRead])
def get_set_flashcards(set_id: UUID, session: Session = Depends(get_session)):
    statement = select(Flashcard).where(Flashcard.flashcard_set_id == set_id)
    flashcards = session.exec(statement).all()
    return flashcards
