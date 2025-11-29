from pydantic import BaseModel
from api.models import UserBase, TopicBase, FlashcardSetBase, FlashcardBase, UploadedFileBase
from uuid import UUID
from datetime import datetime


class UserCreate(BaseModel):
    username: str

class UserRead(UserBase):
    id: int
    created_at: datetime

class TopicCreate(BaseModel):
    title: str

class TopicRead(TopicBase):
    id: UUID
    user_id: int
    created_at: datetime

class FlashcardSetCreate(FlashcardSetBase):
    topic_id: UUID

class FlashcardSetRead(FlashcardSetBase):
    id: UUID
    topic_id: UUID
    created_at: datetime

class FlashcardCreate(FlashcardBase):
    flashcard_set_id: UUID

class FlashcardRead(FlashcardBase):
    id: UUID
    flashcard_set_id: UUID
    created_at: datetime

class UploadedFileRead(UploadedFileBase):
    id: UUID
    user_id: int
    topic_id: UUID
    created_at: datetime
