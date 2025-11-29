from sqlmodel import Session, select
from api.models import UploadedFile, Topic
from fastapi import UploadFile
from typing import List
from uuid import UUID
import os
import shutil
from pathlib import Path
from api.llm_client import get_client


# Use absolute path relative to api folder
API_DIR = Path(__file__).parent.parent
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(API_DIR / "uploads"))

async def upload_files(
    session: Session,
    topic_id: UUID,
    user_id: int,
    files: List[UploadFile]
) -> List[UploadedFile]:
    """Upload files for a topic"""
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
    
    # Generate system prompt based on uploaded files
    try:
        # Get all files for the topic, including the newly uploaded ones
        all_topic_files = get_topic_files(session, topic_id)
        
        file_contents = []
        for db_file in all_topic_files:
            with open(db_file.file_path, "rb") as f:
                content = f.read()
                file_contents.append({
                    "filename": db_file.file_name,
                    "content_type": "application/pdf" if db_file.file_name.endswith(".pdf") else "text/plain",
                    "content": content
                })
        
        client = get_client()
        system_prompt = client.generate_topic_system_prompt(file_contents)
        
        # Update topic
        topic = session.get(Topic, topic_id)
        if topic:
            topic.system_prompt = system_prompt
            session.add(topic)
            
    except Exception as e:
        print(f"Error generating system prompt: {e}")
    
    session.commit()
    
    # Refresh all uploaded files to get their IDs
    for db_file in uploaded_files:
        session.refresh(db_file)
    
    return uploaded_files

def get_topic_files(session: Session, topic_id: UUID) -> List[UploadedFile]:
    """Get all files for a topic"""
    statement = select(UploadedFile).where(UploadedFile.topic_id == topic_id)
    files = session.exec(statement).all()
    return files
