"""Fonctions liées à la base de données."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy import types as satypes
from urllib.parse import quote_plus
from contextlib import contextmanager

from .config import DatabaseConfig


def build_connection_url(cfg: DatabaseConfig) -> str:
    """Construit une URL SQLAlchemy depuis la configuration."""

    return (
        "postgresql+psycopg2://"
        f"{cfg.user}:{quote_plus(cfg.password)}@{cfg.host}:{cfg.port}/{cfg.database}"
    )


def create_pg_engine(cfg: DatabaseConfig) -> Engine:
    """Crée un engine SQLAlchemy configuré pour une connexion distante."""

    return create_engine(
        build_connection_url(cfg),
        connect_args={
            "sslmode": cfg.sslmode,
            "connect_timeout": 5,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 3,
        },
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=5,
    )


@contextmanager
def ensure_connection(engine: Engine):
    """Context manager qui vérifie la connexion avant utilisation."""

    try:
        with engine.connect() as conn:
            conn.execute(text("select 1"))
            yield conn
    except OperationalError as exc:
        raise RuntimeError("Connexion PostgreSQL échouée") from exc


def sql_dtype(tag: str) -> satypes.TypeEngine:
    """Retourne le type SQLAlchemy correspondant au tag détecté."""

    return {
        "DATE": satypes.Date(),
        "DATETIME": satypes.DateTime(),
        "INT": satypes.Integer(),
        "FLOAT": satypes.Float(precision=53),
        "BOOL": satypes.Boolean(),
        "STRING": satypes.Text(),
    }.get(tag, satypes.Text())
