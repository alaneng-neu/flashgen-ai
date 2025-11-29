from sqlmodel import Session, select
from api.models import User
from api.schemas import UserCreate
from fastapi import HTTPException


def create_user(session: Session, user_data: UserCreate) -> User:
    """Create a new user"""
    # Check if username exists
    existing_user = session.exec(select(User).where(User.username == user_data.username)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    db_user = User(username=user_data.username)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user

def get_user(session: Session, user_id: int) -> User:
    """Get user by ID"""
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
