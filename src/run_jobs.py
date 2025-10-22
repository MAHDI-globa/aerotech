# src/run_jobs.py
from __future__ import annotations
from pathlib import Path
import os, importlib, shutil, time, traceback
from datetime import datetime
from dotenv import load_dotenv

from .logging_utils import build_logger, correlation

# --------- CONFIG ---------
JOBS = [
    ("FILE_PATH_FQT", "src.etl_fqt"),              # Qualité trigrammes
    ("FILE_PATH_FNP", "src.etl_fnp"),              # FNP
    ("FILE_PATH_SPA", "src.etl_spa"),              # Suivi Projets AEC
    ("FILE_PATH_RPA", "src.etl_rpa"),              # Relevé pointages AEB
    ("FILE_PATH_CDP", "src.etl_cdp"),              # Conditions de paiement
    ("FILE_PATH_PAAE", "src.etl_paae"),            # Projet AEC
    ("FILE_PATH_CATCLIENTS", "src.etl_catclients") # Catégories clients
]

# Extensions admises uniquement pour les fichiers attendus
ALLOWED_EXTS = {".xlsx", ".xls"}

# Fichiers à ignorer/rejeter directement
TMP_PREFIXES = ("~$",)           # fichiers temporaires Excel
HIDDEN_PREFIXES = (".",)         # fichiers cachés éventuels

ARCHIVE_ROOT_NAME = "Archives"
ENV  = os.getenv("ENV", "dev").lower()
ROOT = Path(__file__).resolve().parent.parent

# délai pour considérer un fichier "stable" (taille inchangée)
READY_GRACE_SEC = float(os.getenv("READY_GRACE_SEC", "2"))

# --------- LOGGING ---------
logger = build_logger("run_jobs", logfile="run_jobs.log")

# --------- ENV LOADING ---------
def load_env_files() -> Path | None:
    """Charge .env.{ENV} (DB) puis .env.files.{ENV} (ou fallback .env.files)."""
    db_env = ROOT / f".env.{ENV}"
    if db_env.exists():
        load_dotenv(db_env)
        logger.debug("DB env loaded: %s", db_env)
    else:
        raise RuntimeError(f".env.{ENV} introuvable: {db_env}")

    files_env = ROOT / f".env.files.{ENV}"
    fallback  = ROOT / ".env.files"
    if files_env.exists():
        load_dotenv(files_env)
        logger.debug("Files env loaded: %s", files_env)
        return files_env
    if fallback.exists():
        load_dotenv(fallback)
        logger.debug("Files env loaded (fallback): %s", fallback)
        return fallback

    logger.info("Aucun fichier .env.files trouvé — rien à faire.")
    return None

# --------- ARCHIVES / REJECTS ---------
def _make_arch_dir(base_dir: Path, kind: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    root = base_dir / ARCHIVE_ROOT_NAME
    root.mkdir(exist_ok=True)
    out = root / f"{kind}_{ts}"
    out.mkdir(exist_ok=True)
    return out

def _unique_path(folder: Path, name: str) -> Path:
    dst = folder / name
    if not dst.exists():
        return dst
    stem, suf = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        cand = folder / f"{stem}__{i}{suf}"
        if not cand.exists():
            return cand
        i += 1

def archive_ok(src: Path) -> None:
    outdir = _make_arch_dir(src.parent, "Loaded")
    dst = _unique_path(outdir, src.name)
    try:
        shutil.move(str(src), str(dst))
        logger.info("Archive OK: %s -> %s", src.name, dst)
    except Exception as e:
        logger.exception("Archive failed for %s -> %s", src, dst)
        # si archive échoue, on tente de rejeter avec la raison
        try:
            reject_file(src, f"Archive failed: {e}")
        except Exception:
            pass

def reject_file(src: Path, reason: str) -> None:
    outdir = _make_arch_dir(src.parent, "Rejects")
    dst = _unique_path(outdir, src.name)
    try:
        shutil.move(str(src), str(dst))
    except Exception:
        logger.exception("Reject move failed for %s", src)
        return
    # raison: nom unique pour éviter collision
    reason_name = _unique_path(outdir, f"{dst.stem}__reason.txt").name
    (outdir / reason_name).write_text(reason.strip() + "\n", encoding="utf-8")
    logger.warning("Reject: %s -> %s | reason=%s", src.name, dst, reason.splitlines()[0])

# --------- FILE READINESS ---------
def is_temp_or_hidden(p: Path) -> bool:
    base = p.name
    return base.startswith(TMP_PREFIXES) or base.startswith(HIDDEN_PREFIXES)

def is_zero_byte(p: Path) -> bool:
    try:
        return p.stat().st_size == 0
    except FileNotFoundError:
        return True  # disparu entre-temps → traite comme invalide

def wait_until_stable(p: Path, grace_sec: float) -> bool:
    """Retourne True si la taille reste stable sur `grace_sec` secondes."""
    try:
        s1 = p.stat().st_size
        time.sleep(grace_sec)
        s2 = p.stat().st_size
        return s1 == s2
    except FileNotFoundError:
        return False

# --------- EXECUTE ETL ---------
def call_module_main(module_path: str) -> None:
    mod = importlib.import_module(module_path)
    fn = getattr(mod, "main", None)
    if not callable(fn):
        raise RuntimeError(f"{module_path} n’a pas de main()")
    fn()

# --------- CORE RUNNER ---------
def run() -> None:
    files_env_loaded = load_env_files()
    if not files_env_loaded:
        return

    with correlation("run-"):  # corr_id partagé dans tous les logs de ce run
        # Mapping des fichiers attendus -> (env_key, module)
        expected: dict[Path, tuple[str, str]] = {}
        watched_dirs: set[Path] = set()

        for env_key, module_path in JOBS:
            file_path = os.getenv(env_key, "").strip()
            if not file_path:
                logger.debug("Var %s non définie — skip", env_key)
                continue
            full = Path(file_path).expanduser().resolve()
            expected[full] = (env_key, module_path)
            watched_dirs.add(full.parent)

        if not expected:
            logger.info("Aucun fichier attendu (.env.files non renseigné). Fin.")
            return

        logger.info("ENV=%s | files-env=%s", ENV, files_env_loaded.name)
        for d in sorted(watched_dirs):
            logger.debug("[scan] %s", d)

        processed_any = False
        cnt_ok = 0
        cnt_reject = 0

        # Parcourt TOUS les fichiers présents dans les dossiers surveillés
        for folder in sorted(watched_dirs):
            if not folder.exists():
                logger.warning("Dossier introuvable: %s", folder)
                continue

            for src in sorted(folder.iterdir()):
                if not src.is_file():
                    continue

                # ignore/rejette les fichiers temporaires ou cachés
                if is_temp_or_hidden(src):
                    reject_file(src, "Fichier temporaire/caché")
                    processed_any = True
                    cnt_reject += 1
                    continue

                # rejette les fichiers vides
                if is_zero_byte(src):
                    reject_file(src, "Fichier vide (0 octet)")
                    processed_any = True
                    cnt_reject += 1
                    continue

                src_full = src.resolve()

                # 1) Fichier exactement attendu ?
                if src_full in expected:
                    env_key, module_path = expected[src_full]

                    # extension autorisée ?
                    if src_full.suffix.lower() not in ALLOWED_EXTS:
                        reject_file(src_full, f"Extension non autorisée pour un fichier attendu: {src_full.suffix}")
                        processed_any = True
                        cnt_reject += 1
                        continue

                    # fichier stable ?
                    if not wait_until_stable(src_full, READY_GRACE_SEC):
                        # si encore en cours de copie/écriture, on réessaiera au prochain run
                        logger.info("Fichier pas encore stable, on le traitera plus tard: %s", src_full)
                        continue

                    processed_any = True
                    logger.info("[job] match %s -> %s | file=%s", env_key, module_path, src_full)

                    t0 = time.perf_counter()
                    try:
                        # (important) réassigne la variable env au chemin réellement vu
                        os.environ[env_key] = str(src_full)
                        call_module_main(module_path)
                        dt = time.perf_counter() - t0
                        logger.info("[ok] %s terminé en %.1fs", module_path, dt)
                        archive_ok(src_full)
                        cnt_ok += 1
                    except Exception as e:
                        tb = traceback.format_exc(limit=5)
                        reject_file(src_full, f"ETL failed: {e}\n{tb}")
                        logger.exception("ETL failed for %s", src_full)
                        cnt_reject += 1
                    continue

                # 2) Fichier non répertorié → rejet systématique
                reject_file(src_full, "Fichier non répertorié dans .env.files.* (chemin inattendu)")
                processed_any = True
                cnt_reject += 1

        if not processed_any:
            logger.info("Aucun fichier à traiter. Fin.")
        else:
            logger.info("Résumé: ok=%d | reject=%d", cnt_ok, cnt_reject)

if __name__ == "__main__":
    run()
