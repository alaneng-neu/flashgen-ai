from sqlmodel import Session, select
from models import FlashcardSet, Flashcard, FlashcardTypeEnum
from schemas import FlashcardSetCreate
from typing import List
from uuid import UUID


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
    system_prompt: str = None
) -> List[Flashcard]:
    """Generate flashcards from file contents using AI/LLM"""
    
    # TODO: Generate flashcards using:
    # - system_prompt (context about the topic) -> from topic.system_prompt
    # - file_contents (list of dicts with filename, content_type, and content bytes)
    # - AI/LLM integration to parse content and create flashcards
    #
    # Example pseudocode:
    # generated_flashcards = generate_with_llm(
    #     system_prompt=system_prompt,
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
