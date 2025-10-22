# src/logging_utils.py
from __future__ import annotations
import os, sys, logging, getpass, uuid
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from contextlib import contextmanager

# =========================
#  LECTURE STRICTE DU .env
# =========================
def _get_env_required(name: str) -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        raise RuntimeError(
            f"Variable d'environnement manquante: {name}. "
            f"Définis-la dans ton .env (ex: {name}=...)."
        )
    return str(v).strip()

def _get_env_int_required(name: str, min_value: int | None = None) -> int:
    raw = _get_env_required(name)
    try:
        val = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} doit être un entier valide (valeur actuelle: {raw!r}).")
    if min_value is not None and val < min_value:
        raise RuntimeError(f"{name} doit être >= {min_value} (valeur actuelle: {val}).")
    return val

# ==== Vars REQUISES depuis .env ====
#   LOG_DIR=logs
#   LOG_LEVEL=INFO
#   LOG_ROTATE=time      # "time" | "size" | "none"
#   LOG_BACKUP_COUNT=365 # rétention (nb de fichiers historiques)
#   LOG_MAX_BYTES=5242880  # requis SEULEMENT si LOG_ROTATE=size
LOG_DIR        = Path(_get_env_required("LOG_DIR"))
LOG_LEVEL_STR  = _get_env_required("LOG_LEVEL").upper()

LOG_ROTATE     = _get_env_required("LOG_ROTATE").lower()   # "time" | "size" | "none"
if LOG_ROTATE not in {"time", "size", "none"}:
    raise RuntimeError(
        "LOG_ROTATE invalide. Valeurs autorisées: 'time' (rotation quotidienne), "
        "'size' (rotation par taille), 'none' (pas de rotation)."
    )

LOG_BACKUP_COUNT = _get_env_int_required("LOG_BACKUP_COUNT", min_value=0)

# requis uniquement si rotation par taille
if LOG_ROTATE == "size":
    LOG_MAX_BYTES = _get_env_int_required("LOG_MAX_BYTES", min_value=1)
else:
    LOG_MAX_BYTES = None

# ==== Mode unifié (un seul fichier pour tous les loggers) ====
#   LOG_UNIFIED=1|0
#   LOG_FILE_NAME=extracteur.log
LOG_UNIFIED = os.getenv("LOG_UNIFIED", "0").strip().lower() in {"1", "true", "yes"}
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", "extracteur.log").strip()  # utilisé si LOG_UNIFIED=1

# ============ Helpers =============
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _current_env() -> str:
    return os.getenv("ENV", "dev").lower()

def _current_user() -> str:
    return os.getenv("LOADED_BY") or getpass.getuser()

def _current_corr() -> str:
    return os.getenv("CORR_ID", "")

# ============ Logger =============
def build_logger(name: str, logfile: str | None = None) -> logging.Logger:
    """
    Logger simple & strict:
      - console + fichier (append)
      - rotation: "time" | "size" | "none"
      - LOG_UNIFIED=1 force tous les loggers à écrire dans LOG_FILE_NAME
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # déjà configuré
        return logger

    # niveau
    try:
        level = getattr(logging, LOG_LEVEL_STR)
    except AttributeError:
        raise RuntimeError(f"LOG_LEVEL invalide: {LOG_LEVEL_STR!r} (ex: DEBUG, INFO, WARNING, ERROR).")
    logger.setLevel(level)

    # format
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] env=%(env)s user=%(user)s mod=%(name)s corr=%(corr)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    class ContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.env  = _current_env()
            record.user = _current_user()
            record.corr = _current_corr()
            return True

    ctx = ContextFilter()

    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    sh.addFilter(ctx)
    logger.addHandler(sh)

    # Fichier (append)
    _ensure_dir(LOG_DIR)
    if LOG_UNIFIED:
        # forcer un seul fichier pour tout le monde
        effective_logfile = LOG_FILE_NAME
    else:
        # fichier par module (comportement actuel)
        effective_logfile = logfile or f"{name}.log"

    log_path = LOG_DIR / effective_logfile

    if LOG_ROTATE == "size":
        fh = RotatingFileHandler(
            filename=str(log_path),
            mode="a",
            maxBytes=LOG_MAX_BYTES,        # non-None ici
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
            delay=True
        )
    elif LOG_ROTATE == "time":
        fh = TimedRotatingFileHandler(
            filename=str(log_path),
            when="midnight",
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
            utc=False,
            delay=True
        )
    else:  # "none"
        fh = logging.FileHandler(
            filename=str(log_path),
            mode="a",
            encoding="utf-8",
            delay=True
        )

    fh.setLevel(level)
    fh.setFormatter(fmt)
    fh.addFilter(ctx)
    logger.addHandler(fh)

    logger.propagate = False
    return logger

def get_logger(name: str, logfile: str | None = None) -> logging.Logger:
    return build_logger(name, logfile)

def set_loaded_by(user: str) -> None:
    os.environ["LOADED_BY"] = str(user)

@contextmanager
def correlation(prefix: str = ""):
    old = os.environ.get("CORR_ID")
    corr = f"{prefix}{uuid.uuid4().hex[:8]}"
    os.environ["CORR_ID"] = corr
    try:
        yield corr
    finally:
        if old is None:
            os.environ.pop("CORR_ID", None)
        else:
            os.environ["CORR_ID"] = old
