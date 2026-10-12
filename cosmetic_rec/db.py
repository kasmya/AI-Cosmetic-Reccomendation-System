"""Persistence for recommendation events and user feedback (SQLAlchemy; SQLite by default, Postgres in production).

Set DATABASE_URL (for example postgresql+psycopg://user:pass@host:5432/cosmetic) to use Postgres. The default is a
SQLite file under var/. Tables are created at startup (create_all); there is no migration tool yet.

Privacy: only the skin type, concern names, budget and the products shown are stored. No photos, no free text,
no IP addresses, no user accounts. The request id is a random UUID the server hands to the client.
"""
import logging
import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

log = logging.getLogger("cosmetic_rec.db")
VAR_DIR = Path(__file__).resolve().parent.parent / "var"


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class RecommendationEvent(Base):
    """One /consult call: who it was for (skin type and concerns only) and what was shown."""
    __tablename__ = "recommendation_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    model_version: Mapped[str] = mapped_column(String(64), default="unversioned")
    skin_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    concerns: Mapped[list] = mapped_column(JSON, default=list)
    budget: Mapped[float | None] = mapped_column(Float, nullable=True)
    shown: Mapped[list] = mapped_column(JSON, default=list)        # [{product_id, score, p_like, verdict}]


class Feedback(Base):
    """A thumbs up / down on one product, optionally tied to the consultation that showed it."""
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("event_id", "product_id", name="one_vote_per_product_per_event"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str | None] = mapped_column(ForeignKey("recommendation_events.id"), nullable=True, index=True)
    product_id: Mapped[int] = mapped_column(Integer, index=True)
    helpful: Mapped[bool] = mapped_column(Boolean)
    skin_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


def normalise_url(url: str) -> str:
    """Hosts (Neon, Render, Heroku...) hand out postgres:// or postgresql:// URLs; SQLAlchemy needs the driver named."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return normalise_url(url)
    VAR_DIR.mkdir(exist_ok=True)
    return f"sqlite:///{VAR_DIR / 'app.db'}"


class Database:
    def __init__(self, url: str | None = None):
        self.url = normalise_url(url) if url else database_url()
        self.engine: Engine = create_engine(self.url, pool_pre_ping=True, future=True)
        self._session = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name

    @contextmanager
    def session(self):
        s: Session = self._session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ---- writes ----
    def log_event(self, model_version: str, skin_type, concerns: list[str], budget, shown: list[dict]) -> str | None:
        """Record what a consultation showed. Never raises: logging must not break a recommendation."""
        try:
            with self.session() as s:
                ev = RecommendationEvent(model_version=model_version, skin_type=skin_type, concerns=list(concerns),
                                         budget=budget, shown=shown)
                s.add(ev)
                s.flush()
                return ev.id
        except Exception:
            log.exception("could not log recommendation event")
            return None

    def add_feedback(self, product_id: int, helpful: bool, event_id: str | None = None) -> dict:
        """Store a vote; a second vote on the same product in the same consultation replaces the first."""
        with self.session() as s:
            skin_type = None
            if event_id:
                ev = s.get(RecommendationEvent, event_id)
                if ev is None:
                    raise KeyError(event_id)
                skin_type = ev.skin_type
                existing = s.scalar(select(Feedback).where(Feedback.event_id == event_id,
                                                           Feedback.product_id == product_id))
                if existing:
                    existing.helpful = helpful
                    return {"id": existing.id, "updated": True}
            fb = Feedback(event_id=event_id, product_id=product_id, helpful=helpful, skin_type=skin_type)
            s.add(fb)
            s.flush()
            return {"id": fb.id, "updated": False}

    # ---- reads ----
    def feedback_rows(self) -> list[dict]:
        with self.session() as s:
            return [{"product_id": f.product_id, "helpful": f.helpful, "skin_type": f.skin_type,
                     "event_id": f.event_id, "created_at": f.created_at} for f in s.scalars(select(Feedback))]

    def counts(self) -> dict:
        with self.session() as s:
            return {"events": s.scalar(select(func.count()).select_from(RecommendationEvent)),
                    "feedback": s.scalar(select(func.count()).select_from(Feedback)),
                    "helpful": s.scalar(select(func.count()).select_from(Feedback).where(Feedback.helpful.is_(True)))}

    def recent_events(self, limit: int = 2000) -> list[dict]:
        with self.session() as s:
            rows = s.scalars(select(RecommendationEvent).order_by(RecommendationEvent.created_at.desc()).limit(limit))
            return [{"skin_type": e.skin_type, "shown": e.shown, "model_version": e.model_version,
                     "created_at": e.created_at} for e in rows]
