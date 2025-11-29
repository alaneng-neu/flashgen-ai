from sqlmodel import Session, select
from api.models import FlashcardSet, Flashcard
from api.schemas import FlashcardSetCreate
from typing import List
from uuid import UUID
from api.llm_client import get_client


def get_flashcard_set(session: Session, set_id: UUID) -> FlashcardSet:
    """Get flashcard set by ID"""
    from fastapi import HTTPException
    flashcard_set = session.get(FlashcardSet, set_id)
    if not flashcard_set:
        raise HTTPException(status_code=404, detail="Flashcard set not found")
    return flashcard_set

def get_topic_flashcard_sets(session: Session, topic_id: UUID) -> List[FlashcardSet]:
    """Get all flashcard sets for a topic"""
    statement = select(FlashcardSet).where(FlashcardSet.topic_id == topic_id)
    sets = session.exec(statement).all()
    return sets

async def generate_flashcards(
    session: Session,
    flashcard_set_id: UUID,
    file_contents: List[dict],
    system_prompt: str = None,
    num_flashcards: int = 10
) -> List[Flashcard]:
    """Generate flashcards from file contents using AI/LLM"""
    
    client = get_client()
    generated_flashcards = await client.generate_flashcards_from_files(
        file_contents=file_contents,
        system_prompt=system_prompt,
        num_flashcards=num_flashcards
    )
    
    # Save generated flashcards to database
    db_flashcards = []
    for fc_data in generated_flashcards:
        db_flashcard = Flashcard(
            term=fc_data["term"],
            definition=fc_data["definition"],
            flashcard_type=fc_data["flashcard_type"],
            flashcard_set_id=flashcard_set_id
        )
        session.add(db_flashcard)
        db_flashcards.append(db_flashcard)
    
    session.commit()
    
    # Refresh all flashcards to get their IDs
    for db_flashcard in db_flashcards:
        session.refresh(db_flashcard)
    
    return db_flashcards

def get_set_flashcards(session: Session, set_id: UUID) -> List[Flashcard]:
    """Get all flashcards for a set"""
    statement = select(Flashcard).where(Flashcard.flashcard_set_id == set_id)
    flashcards = session.exec(statement).all()
    return flashcards

def create_flashcard_set(session: Session, topic_id: UUID, set_data: FlashcardSetCreate) -> FlashcardSet:
    """Create a new flashcard set"""
    db_set = FlashcardSet.model_validate(set_data)
    db_set.topic_id = topic_id
    session.add(db_set)
    session.commit()
    session.refresh(db_set)
    return db_set
