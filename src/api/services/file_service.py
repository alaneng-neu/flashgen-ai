from sqlmodel import Session, select
from models import UploadedFile
from fastapi import UploadFile
from typing import List
from uuid import UUID
import os
import shutil
from pathlib import Path


UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

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
    
    # TODO: Generate system prompt based on uploaded files
    # Example: Analyze file contents, extract key concepts, generate prompt
    # topic.system_prompt = generate_system_prompt(uploaded_files)
    
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
