# src/watch_jobs.py
from __future__ import annotations
from pathlib import Path
import os, sys, time, subprocess
from datetime import datetime
from dotenv import load_dotenv

# --------- charger .env AVANT le logger ---------
ROOT = Path(__file__).resolve().parent.parent
ENV  = os.getenv("ENV", "dev").lower()

env_file = ROOT / f".env.{ENV}"
if not env_file.exists():
    raise RuntimeError(f".env.{ENV} introuvable: {env_file}")
load_dotenv(env_file)

# .env.files.<ENV> (utile si tu veux aussi que WATCH_POLL_INTERVAL soit dans .env.*)
files_env = ROOT / f".env.files.{ENV}"
fallback_files_env = ROOT / ".env.files"
if files_env.exists():
    load_dotenv(files_env)
elif fallback_files_env.exists():
    load_dotenv(fallback_files_env)

# maintenant que .env est chargé, on peut importer le logger
from .logging_utils import get_logger, correlation  # logging unifié

# fichiers à ignorer (temporaires courants)
IGNORED_PREFIXES = ("~$",)
IGNORED_SUFFIXES = (".tmp", ".part", ".crdownload", ".swp", ".lock")

# ⚠️ un SEUL fichier de logs partagé avec run_jobs
logger = get_logger("run_jobs", logfile="run_jobs.log")

def collect_watch_dirs() -> list[Path]:
    """
    Récupère toutes les variables FILE_PATH_* et retourne l'ensemble
    des dossiers parents (dédupliqués) à surveiller.
    """
    dirs: set[Path] = set()
    for k, v in os.environ.items():
        if not k.startswith("FILE_PATH_"):
            continue
        p = (v or "").strip()
        if not p:
            continue
        try:
            parent = Path(p).expanduser().resolve().parent
            dirs.add(parent)
        except Exception:
            continue
    return sorted(dirs)

def should_ignore(path: Path) -> bool:
    name = path.name
    if any(name.startswith(pref) for pref in IGNORED_PREFIXES):
        return True
    if any(name.lower().endswith(suf) for suf in IGNORED_SUFFIXES):
        return True
    return False

def run_jobs_once():
    """Lance l’orchestrateur en sous-process (hérite de l'env)."""
    cmd = [sys.executable, "-m", "src.run_jobs"]
    logger.info("[run] %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            logger.warning("run_jobs a retourné le code %s", res.returncode)
    except Exception:
        logger.exception("Échec exécution run_jobs")

def main():
    # relire la fréquence de scan après chargement des .env
    try:
        poll_interval = int(os.getenv("WATCH_POLL_INTERVAL", "5"))
    except ValueError:
        poll_interval = 5

    watch_dirs = collect_watch_dirs()

    with correlation("watch-"):  # un corr_id par session de watch
        logger.info("ENV=%s | env_file=%s | poll=%ss", ENV, env_file.name, poll_interval)
        if not watch_dirs:
            logger.info("Aucun FILE_PATH_* défini → aucun dossier à surveiller. Fin.")
            return

        logger.info("[watch] Surveillance démarrée (Ctrl+C pour arrêter)")
        for d in watch_dirs:
            logger.info("[watch] dir: %s", d)

        seen_mtime: dict[str, float] = {}

        try:
            while True:
                any_change = False
                for d in watch_dirs:
                    if not d.exists():
                        logger.warning("Dossier introuvable: %s", d)
                        continue
                    try:
                        for f in d.iterdir():
                            if not f.is_file() or should_ignore(f):
                                continue
                            key = str(f.resolve())
                            try:
                                mtime = f.stat().st_mtime
                            except FileNotFoundError:
                                seen_mtime.pop(key, None)
                                continue
                            prev = seen_mtime.get(key)
                            if prev is None or mtime > prev:
                                logger.info("[detected] %s (new/updated)", f)
                                seen_mtime[key] = mtime
                                any_change = True

                        # purge les fichiers disparus du cache
                        current_keys = {
                            str(p.resolve()) for p in d.iterdir()
                            if p.is_file() and not should_ignore(p)
                        }
                        stale = [
                            k for k in list(seen_mtime.keys())
                            if k.startswith(str(d.resolve())) and k not in current_keys
                        ]
                        for k in stale:
                            seen_mtime.pop(k, None)

                    except Exception:
                        logger.exception("Scan échoué sur %s", d)

                if any_change:
                    run_jobs_once()
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Arrêt demandé. Bye.")

if __name__ == "__main__":
    main()
