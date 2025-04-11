from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import timezone, datetime
import os

Base = declarative_base()

class TaskDB(Base):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    estimated_duration = Column(Integer, nullable=False)
    required_focus = Column(Float, default=0.5)
    category = Column(String, default="general")
    flexibility = Column(Float, default=0.5)
    priority = Column(Integer, default=3)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    recurrence = Column(String, default="none")

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
