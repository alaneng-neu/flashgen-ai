from fastapi import APIRouter, Depends, status
from sqlmodel import Session
from database import get_session
from schemas import TopicCreate, TopicRead
from services import user_service, topic_service
from typing import List
from uuid import UUID


router = APIRouter(tags=["topics"])

@router.post("/users/{user_id}/topics", response_model=TopicRead, status_code=status.HTTP_201_CREATED)
def create_topic(user_id: int, topic: TopicCreate, session: Session = Depends(get_session)):
    """Create a new topic for a user"""
    # Verify user exists
    user_service.get_user(session, user_id)
    
    return topic_service.create_topic(session, user_id, topic)

@router.get("/topics/{topic_id}", response_model=TopicRead)
def get_topic(topic_id: UUID, session: Session = Depends(get_session)):
    """Get topic by ID"""
    return topic_service.get_topic(session, topic_id)

@router.get("/users/{user_id}/topics", response_model=List[TopicRead])
def get_user_topics(user_id: int, session: Session = Depends(get_session)):
    """Get all topics for a user"""
    return topic_service.get_user_topics(session, user_id)
