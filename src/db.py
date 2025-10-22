# src/db.py
from __future__ import annotations
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

from .logging_utils import get_logger  # ← logging centralisé

# --------- ENV ---------
ROOT = Path(__file__).resolve().parent.parent
ENV = os.getenv("ENV", "dev").lower()           # dev | uat | prod
ENV_FILE = ROOT / f".env.{ENV}"
if not ENV_FILE.exists():
    raise RuntimeError(f"Fichier {ENV_FILE} introuvable. Crée .env.{ENV}")
load_dotenv(ENV_FILE)

# --------- LOGGER ---------
logger = get_logger(__name__, logfile="db.log")

def _mask(s: str | None) -> str:
    """Masque les secrets dans les logs (ex. mot de passe)."""
    if not s:
        return ""
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]

def get_engine(prefix: str = "PG_") -> Engine:
    """
    Construit un Engine SQLAlchemy à partir des variables d'env:
      {prefix}HOST, {prefix}PORT, {prefix}DB, {prefix}USER, {prefix}PASSWORD, {prefix}SSLMODE
    """
    host = os.getenv(prefix + "HOST")
    port = os.getenv(prefix + "PORT", "5432")
    db   = os.getenv(prefix + "DB")
    user = os.getenv(prefix + "USER")
    pwd  = os.getenv(prefix + "PASSWORD")
    ssl  = os.getenv(prefix + "SSLMODE", "require")

    missing = [k for k, v in {
        f"{prefix}HOST": host,
        f"{prefix}DB": db,
        f"{prefix}USER": user,
        f"{prefix}PASSWORD": pwd
    }.items() if not v]

    if missing:
        logger.error("Variables manquantes pour la connexion DB: %s (env=%s)", ", ".join(missing), ENV)
        raise RuntimeError(f"Variables manquantes: {', '.join(missing)} (env={ENV})")

    logger.info(
        "DB target resolved | env=%s host=%s db=%s user=%s ssl=%s",
        ENV, host, db, user, ssl
    )
    # NB: on NE LOG PAS le mot de passe, et on masque si besoin:
    logger.debug("Connexion params (debug) | host=%s port=%s db=%s user=%s pwd=%s ssl=%s",
                 host, port, db, user, _mask(pwd), ssl)

    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode={ssl}"
    eng = create_engine(url, pool_pre_ping=True, pool_recycle=180)
    logger.debug("SQLAlchemy engine créé (pool_pre_ping=True, pool_recycle=180)")
    return eng

def check_connection() -> str:
    """Ouvre une connexion, exécute un ping, renvoie un message lisible, et loggue le résultat."""
    try:
        eng = get_engine()
        with eng.connect() as conn:
            ver = conn.exec_driver_sql("SHOW server_version;").scalar()
            conn.execute(text("SELECT 1"))
        msg = f"Connexion OK — Postgres {ver}"
        logger.info(msg)
        return msg
    except Exception as e:
        logger.exception("Échec de la vérification de connexion DB")
        # On renvoie aussi l’erreur pour l’appelant, au besoin
        raise
