from fastapi import APIRouter, Depends, status
from sqlmodel import Session
from database import get_session
from schemas import UserCreate, UserRead
from services import user_service


router = APIRouter(prefix="/users", tags=["users"])

@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, session: Session = Depends(get_session)):
    """Create a new user"""
    return user_service.create_user(session, user)

@router.get("/{user_id}", response_model=UserRead)
def get_user(user_id: int, session: Session = Depends(get_session)):
    """Get user by ID"""
    return user_service.get_user(session, user_id)
