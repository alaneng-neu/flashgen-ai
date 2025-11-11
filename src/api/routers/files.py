from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlmodel import Session
from database import get_session
from schemas import UploadedFileRead
from services import user_service, topic_service, file_service
from typing import List
from uuid import UUID


router = APIRouter(tags=["files"])

@router.post("/topics/{topic_id}/upload", response_model=List[UploadedFileRead], status_code=status.HTTP_201_CREATED)
async def upload_files(
    topic_id: UUID,
    user_id: int = Form(...),
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session)
):
    """Upload files to a topic"""
    # Verify user exists
    user_service.get_user(session, user_id)
    
    # Verify topic exists and belongs to user
    topic = topic_service.get_topic(session, topic_id)
    if topic.user_id != user_id:
        raise HTTPException(status_code=403, detail="Topic does not belong to user")
    
    return await file_service.upload_files(session, topic_id, user_id, files)

@router.get("/topics/{topic_id}/files", response_model=List[UploadedFileRead])
def get_topic_files(topic_id: UUID, session: Session = Depends(get_session)):
    """Get all files for a topic"""
    return file_service.get_topic_files(session, topic_id)
