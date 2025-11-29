from sqlmodel import create_engine, SQLModel, Session
from typing import Generator
import os
from pathlib import Path
from dotenv import load_dotenv


# Load .env from the api directory
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session