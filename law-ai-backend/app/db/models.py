from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, String, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    cases = relationship("Case", back_populates="user", cascade="all, delete-orphan")


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(512), nullable=True)
    text = Column(Text, nullable=False)
    case_summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="cases")
    qa_items = relationship("QuestionAnswer", back_populates="case", cascade="all, delete-orphan")


class QuestionAnswer(Base):
    __tablename__ = "question_answers"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # The user's input
    question = Column(Text, nullable=False)
    
    # The AI's response
    answer = Column(Text, nullable=False)
    
    # The category (e.g., 'property_division', 'children_parenting')
    # Helps filter history for topic-specific reasoning
    topic = Column(String(50), nullable=True)
    
    # Store the RAG source nodes (file names, scores, and text snippets)
    # Storing this as JSON allows for easy front-end citation rendering
    sources = Column(JSON, nullable=True)

    # A snapshot of the case summary section used at the time of the answer
    # Important because the case summary might change over time
    context_snapshot = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    case = relationship("Case", back_populates="qa_items")
