from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from enum import Enum
from datetime import datetime, timezone
from uuid import UUID, uuid4


class FlashcardTypeEnum(str, Enum):
    TERM_DEFINITION = "term_definition"
    FILL_IN_BLANK = "fill_in_blank"
    LIST_STAGES = "list_stages"
    QUESTION_TO_ANSWER = "question_to_answer"
    EXAMPLE_TO_CONCEPT = "example_to_concept"
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"


# Base models with shared fields
class TimestampMixin(SQLModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# User model
class UserBase(SQLModel):
    username: str = Field(unique=True, index=True)

class User(UserBase, TimestampMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Relationships
    topics: List["Topic"] = Relationship(back_populates="user")
    uploaded_files: List["UploadedFile"] = Relationship(back_populates="user")


# Topic model
class TopicBase(SQLModel):
    title: str
    system_prompt: Optional[str] = None

class Topic(TopicBase, TimestampMixin, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    
    # Relationships
    user: User = Relationship(back_populates="topics")
    flashcard_sets: List["FlashcardSet"] = Relationship(back_populates="topic")
    uploaded_files: List["UploadedFile"] = Relationship(back_populates="topic")


# FlashcardSet model
class FlashcardSetBase(SQLModel):
    title: str

class FlashcardSet(FlashcardSetBase, TimestampMixin, table=True):
    __tablename__ = "flashcard_set"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    topic_id: UUID = Field(foreign_key="topic.id")
    
    # Relationships
    topic: Topic = Relationship(back_populates="flashcard_sets")
    flashcards: List["Flashcard"] = Relationship(back_populates="flashcard_set")


# Flashcard model
class FlashcardBase(SQLModel):
    term: str
    definition: str
    flashcard_type: FlashcardTypeEnum = Field(default=FlashcardTypeEnum.TERM_DEFINITION)

class Flashcard(FlashcardBase, TimestampMixin, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    flashcard_set_id: UUID = Field(foreign_key="flashcard_set.id")
    
    # Relationships
    flashcard_set: FlashcardSet = Relationship(back_populates="flashcards")


# UploadedFile model
class UploadedFileBase(SQLModel):
    file_name: str
    file_path: str
    file_size: Optional[int] = None

class UploadedFile(UploadedFileBase, TimestampMixin, table=True):
    __tablename__ = "uploaded_file"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    topic_id: UUID = Field(foreign_key="topic.id")
    
    # Relationships
    user: User = Relationship(back_populates="uploaded_files")
    topic: Topic = Relationship(back_populates="uploaded_files")
