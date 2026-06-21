"""
database.py
---
Create predictions database for monitoring.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from datetime import datetime, timezone

# Create connexion with database
DATABASE_URL = "sqlite:///./predictions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """SQLAlchemy model representing a single anomaly detection prediction."""

    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    category = Column(String(100), nullable=False, index=True)
    filename = Column(String(200), nullable=False)
    score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    verdict = Column(String(100), nullable=False, index=True)
    inference_time = Column(Float, nullable=False)


# Create the table in the database if it doesn't already exist
Base.metadata.create_all(bind=engine)


def get_db():
    """Open a local database session and close it once the request is done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
