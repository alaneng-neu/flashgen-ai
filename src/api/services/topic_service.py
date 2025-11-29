from sqlmodel import Session, select
from api.models import Topic
from api.schemas import TopicCreate
from fastapi import HTTPException
from typing import List
from uuid import UUID


def create_topic(session: Session, user_id: int, topic_data: TopicCreate) -> Topic:
    """Create a new topic for a user"""
    db_topic = Topic(title=topic_data.title, user_id=user_id)
    session.add(db_topic)
    session.commit()
    session.refresh(db_topic)
    return db_topic

def get_topic(session: Session, topic_id: UUID) -> Topic:
    """Get topic by ID"""
    topic = session.get(Topic, topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    return topic

def get_user_topics(session: Session, user_id: int) -> List[Topic]:
    """Get all topics for a user"""
    statement = select(Topic).where(Topic.user_id == user_id)
    topics = session.exec(statement).all()
    return topics
