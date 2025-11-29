from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlmodel import Session
from api.database import get_session
from api.schemas import FlashcardSetCreate, FlashcardSetRead, FlashcardRead
from api.services import user_service, topic_service, flashcard_service
from typing import List
from uuid import UUID


router = APIRouter(tags=["flashcards"])

@router.get("/flashcard-sets/{set_id}", response_model=FlashcardSetRead)
def get_flashcard_set(set_id: UUID, session: Session = Depends(get_session)):
    """Get flashcard set by ID"""
    return flashcard_service.get_flashcard_set(session, set_id)

@router.get("/topics/{topic_id}/flashcard-sets", response_model=List[FlashcardSetRead])
def get_topic_flashcard_sets(topic_id: UUID, session: Session = Depends(get_session)):
    """Get all flashcard sets for a topic"""
    return flashcard_service.get_topic_flashcard_sets(session, topic_id)

@router.post("/topics/{topic_id}/generate-flashcards", response_model=List[FlashcardRead], status_code=status.HTTP_201_CREATED)
async def generate_flashcard_set(
    topic_id: UUID,
    user_id: int = Form(...),
    title: str = Form(...),
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session)
):
    """Generate flashcards from uploaded files"""
    # Verify user exists
    user_service.get_user(session, user_id)
    
    # Verify topic exists and belongs to user
    topic = topic_service.get_topic(session, topic_id)
    if topic.user_id != user_id:
        raise HTTPException(status_code=403, detail="Topic does not belong to user")
    
    # Create the flashcard set
    flashcard_set_data = FlashcardSetCreate(title=title)
    db_set = flashcard_service.create_flashcard_set(session, topic_id, flashcard_set_data)
    
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
    
    # Generate flashcards using the service
    flashcards = await flashcard_service.generate_flashcards(
        session,
        db_set.id,
        file_contents,
        system_prompt=topic.system_prompt
    )
    
    return flashcards

@router.get("/flashcard-sets/{set_id}/flashcards", response_model=List[FlashcardRead])
def get_set_flashcards(set_id: UUID, session: Session = Depends(get_session)):
    """Get all flashcards for a set"""
    return flashcard_service.get_set_flashcards(session, set_id)
