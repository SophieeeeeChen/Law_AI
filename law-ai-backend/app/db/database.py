from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import Config


def _get_connect_args(db_url: str):
    if db_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


engine = create_engine(
    Config.DATABASE_URL,
    connect_args=_get_connect_args(Config.DATABASE_URL),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    from app.db.models import User, Case, QuestionAnswer  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_case_summary_column()


def _ensure_case_summary_column():
    if not Config.DATABASE_URL.startswith("sqlite"):
        return
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(cases)"))
        cols = [row[1] for row in result.fetchall()]
        if "case_summary" not in cols:
            conn.execute(text("ALTER TABLE cases ADD COLUMN case_summary TEXT"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
