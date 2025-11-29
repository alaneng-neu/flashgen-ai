from sqlmodel import Session, select
from api.models import FlashcardSet, Flashcard
from api.schemas import FlashcardSetCreate
from typing import List
from uuid import UUID
from api.llm_client import get_client
from pathlib import Path

try:
    from quizlet_rag import QuizletRAGPipeline
except ImportError:
    print("Warning: Could not import QuizletRAGPipeline")
    QuizletRAGPipeline = None


# Initialize RAG pipeline lazily or globally
_rag_pipeline = None

def get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None and QuizletRAGPipeline:
        # src/api/services/flashcard_service.py -> src/quizlet_db
        base_dir = Path(__file__).parent.parent.parent
        vector_store_path = base_dir / "quizlet_db"
        
        try:
            _rag_pipeline = QuizletRAGPipeline(vector_store_path=str(vector_store_path))
            # Load existing vector store
            _rag_pipeline.load_existing_vectorstore()
        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
            _rag_pipeline = None
            
    return _rag_pipeline


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
    
    # Generate RAG context
    rag_context = ""
    try:
        pipeline = get_rag_pipeline()
        if pipeline:
            rag_query_obj = client.generate_rag_query(file_contents, system_prompt)
            query_text = rag_query_obj.query
            print(f"RAG Query: {query_text}")
            
            # Query vector store
            results = pipeline.query(query_text, k=5)
            if results:
                rag_context = "Relevant Flashcards from Database:\n"
                for i, doc in enumerate(results):
                    rag_context += f"{i+1}. {doc.page_content}\n"
                print(f"Found {len(results)} RAG documents")
    except Exception as e:
        print(f"Error generating RAG context: {e}")

    generated_flashcards = await client.generate_flashcards_from_files(
        file_contents=file_contents,
        system_prompt=system_prompt,
        rag_context=rag_context,
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
