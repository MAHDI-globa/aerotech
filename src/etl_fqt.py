# src/etl_fqt.py
from __future__ import annotations
from pathlib import Path
import os, getpass, re, unicodedata, warnings
import pandas as pd
import numpy as np

from sqlalchemy import text
from sqlalchemy import types as satypes
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

from .db import get_engine
from .audit import add_audit_columns, audit_dtype_map          # ← AUDIT CENTRALISÉ
from .logging_utils import get_logger, correlation              # ← LOGGING

# ================== CONFIG (local) ================== #
SHOW_SAMPLE    = True
TARGET_SCHEMA  = "public"
IF_EXISTS_MODE = "replace"   # "replace" ou "append"

# Colonnes à forcer en type date (OPTIONNEL)
FORCE_DATE_COLS = {"date_d_attribution", "date_de_retrait"}  # snake_case attendu

# ================== CHARGEMENT DES .env ================== #
ROOT = Path(__file__).resolve().parent.parent
ENV = os.getenv("ENV", "dev").lower()

# 1) BDD
db_env_file = ROOT / f".env.{ENV}"
if not db_env_file.exists():
    raise RuntimeError(f"Fichier d'env BDD introuvable: {db_env_file} (crée .env.{ENV})")
load_dotenv(db_env_file)

# 2) Fichiers
files_env_file = ROOT / f".env.files.{ENV}"
fallback_files_env = ROOT / ".env.files"
if files_env_file.exists():
    load_dotenv(files_env_file); files_env_loaded = files_env_file
elif fallback_files_env.exists():
    load_dotenv(fallback_files_env); files_env_loaded = fallback_files_env
else:
    raise RuntimeError(
        f"Fichier d'env fichiers introuvable: {files_env_file} et {fallback_files_env}.\n"
        f"Crée l'un des deux avec, par ex.:\nFILE_PATH_FQT='C:/chemin/vers/fichier qualité des trigrammes.xlsx'"
    )

FILE_PATH = os.getenv("FILE_PATH_FQT")
if not FILE_PATH:
    raise RuntimeError(f"FILE_PATH_FQT manquant dans {files_env_loaded}")

# ================== LOGGING ================== #
logger = get_logger("etl_fqt", logfile="etl_fqt.log")

# ================== UTILS ================== #
def strip_accents_lower(s):
    if s is None or (isinstance(s, float) and pd.isna(s)): return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()

def snake_id(s):
    s = strip_accents_lower(s)
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"[\s\.\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_") or "col"
    return s

def pg_ident(name: str, maxlen=63) -> str:
    return name[:maxlen]

def parse_number_like(s: pd.Series) -> pd.Series:
    x = s.astype("string")\
         .str.replace("\u00A0", " ", regex=False)\
         .str.replace("\u202F", " ", regex=False)\
         .str.replace(" ", "", regex=False)
    x = x.str.replace(r"(?<=\d)\.(?=\d{3}(?:\D|$))", "", regex=True)\
         .str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")

HEADER_KEYWORDS = (
    "date", "motif", "rédact", "redact", "trig", "nom", "prénom", "prenom",
    "personnel", "site", "retrait", "édition", "edition"
)

def choose_header_row(df: pd.DataFrame, scan=25) -> int:
    limit = min(len(df), scan)
    header_candidate_idx, header_candidate_hits = None, -1
    for i in range(limit):
        row = df.iloc[i].astype(str)
        hits = 0
        for v in row:
            t = strip_accents_lower(v)
            if t and any(k in t for k in HEADER_KEYWORDS):
                hits += 1
        if hits >= 2 and hits > header_candidate_hits:
            header_candidate_hits, header_candidate_idx = hits, i
    if header_candidate_idx is not None:
        return header_candidate_idx
    best, idx = -1, 0
    for i in range(limit):
        row = df.iloc[i].astype(str)
        non_empty = row.map(lambda x: x.strip()!="").sum()
        texty = row.map(lambda x: bool(re.search(r"[A-Za-zÀ-ÿ]", x))).sum()
        score = non_empty*2 + texty
        if score>best: best, idx = score, i
    return idx

def normalize_columns(header_vals):
    cols, seen = [], {}
    for v in header_vals:
        s = snake_id(v) if (v is not None and str(v).strip()!="") else "col"
        seen[s] = seen.get(s,0)+1
        cols.append(s if seen[s]==1 else f"{s}_{seen[s]}")
    return cols

def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    blank_cols = [c for c in df.columns
                  if df[c].isna().all() or df[c].astype(str).str.strip().eq("").all()]
    return df.drop(columns=blank_cols) if blank_cols else df

MAP_TRUE  = {"true","vrai","oui","yes","1","y","o"}
MAP_FALSE = {"false","faux","non","no","0","n"}

DATE_TOKEN_RE = re.compile(
    r"(?:\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})|(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janv|févr|fevr|avr|mai|juin|juil|sept|oct|nov|d[ée]c)",
    re.I
)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?$")
COLNAME_DATE_HINTS = ("date", "dt", "heure", "time")

def infer_col(s: pd.Series):
    ss = s.astype("string").str.strip()
    ss = ss.where(~ss.fillna("").eq("0"), pd.NA)

    if snake_id(s.name) in FORCE_DATE_COLS:
        val = ss.fillna("")
        iso_mask = val.str.match(ISO_DATE_RE)
        if iso_mask.mean() >= 0.5:
            dt = pd.to_datetime(ss, errors="coerce", dayfirst=False)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)
                dt1 = pd.to_datetime(ss, errors="coerce", dayfirst=True,  cache=True)
                dt2 = pd.to_datetime(ss, errors="coerce", dayfirst=False, cache=True)
            dt = dt2 if dt2.notna().sum() > dt1.notna().sum() else dt1
        has_time = (dt.dropna().dt.time.astype(str) != "00:00:00").mean() > 0.2
        return (dt, "DATETIME" if has_time else "DATE")

    iso_mask = ss.fillna("").str.match(ISO_DATE_RE)
    if iso_mask.mean() >= 0.5:
        dt_iso = pd.to_datetime(ss.where(iso_mask), errors="coerce", dayfirst=False)
        if (dt_iso.notna().mean() if len(ss) else 0.0) >= 0.5:
            has_time = (dt_iso.dropna().dt.time.astype(str) != "00:00:00").mean() > 0.2
            return (dt_iso, "DATETIME" if has_time else "DATE")

    as_num = pd.to_numeric(ss, errors="coerce")
    if (as_num.notna().mean() if len(ss) else 0.0) >= 0.9:
        mask = as_num.between(60, 2_950_000)
        parsed = pd.to_datetime(as_num.where(mask), unit="D", origin="1899-12-30", errors="coerce")
        if (parsed.notna().mean() if len(ss) else 0.0) >= 0.7:
            has_time = (parsed.dropna().dt.time.astype(str) != "00:00:00").mean() > 0.2
            return (parsed, "DATETIME" if has_time else "DATE")

    colname_hint = any(h in strip_accents_lower(s.name or "") for h in COLNAME_DATE_HINTS)
    date_token_ratio = ss.fillna("").str.contains(DATE_TOKEN_RE, na=False).mean()
    looks_datey = colname_hint or (date_token_ratio >= 0.30)
    if looks_datey:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)
            dt1 = pd.to_datetime(ss, errors="coerce", dayfirst=True,  cache=True)
            dt2 = pd.to_datetime(ss, errors="coerce", dayfirst=False, cache=True)
        dt = dt2 if dt2.notna().sum() > dt1.notna().sum() else dt1
        ok_abs = dt.notna().mean() if len(ss) else 0.0
        if ok_abs >= (0.50 if colname_hint else 0.70):
            has_time = (dt.dropna().dt.time.astype(str) != "00:00:00").mean() > 0.2
            return (dt, "DATETIME" if has_time else "DATE")

    mb = ss.str.lower().map(lambda x: True if x in MAP_TRUE else (False if x in MAP_FALSE else pd.NA))
    if (mb.notna().mean() if len(ss) else 0) >= 0.98:
        return mb.astype("boolean"), "BOOL"

    nums = parse_number_like(ss)
    if (nums.notna().mean() if len(ss.dropna()) else 0) >= 0.85:
        nz = nums.dropna()
        if len(nz) and np.isclose(nz, np.round(nz)).all():
            return nums.round().astype("Int64"), "INT"
        return nums.astype("Float64"), "FLOAT"

    return ss.astype("string"), "STRING"

def pg_type(tag: str):
    return {
        "DATE": satypes.Date(),
        "DATETIME": satypes.DateTime(),
        "INT": satypes.Integer(),
        "FLOAT": satypes.Float(precision=53),
        "BOOL": satypes.Boolean(),
        "STRING": satypes.Text()
    }.get(tag, satypes.Text())

# ================== MAIN ================== #
def main():
    with correlation("fqt-"):
        xlsx = Path(FILE_PATH)
        logger.info("Start etl_fqt | file=%s env=%s files-env=%s", xlsx, ENV, getattr(files_env_loaded, "name", "?"))
        if not xlsx.exists():
            logger.error("Fichier introuvable: %s", xlsx)
            return

        # Connexion
        try:
            engine = get_engine()
            with engine.connect() as conn:
                ver = conn.execute(text("select version();")).scalar()
                logger.info("DB connectée — %s | ENV=%s | files=%s", ver, ENV, getattr(files_env_loaded, "name", "?"))
        except (OperationalError, RuntimeError):
            logger.exception("Connexion PostgreSQL échouée")
            return

        try:
            sheets = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl", header=None)
            logger.info("Workbook lu: %d feuille(s)", len(sheets))
        except Exception:
            logger.exception("Lecture Excel échouée")
            return

        file_base = snake_id(xlsx.stem)
        done = 0

        for name, raw in sheets.items():
            raw = raw.dropna(how="all", axis=0).dropna(how="all", axis=1)
            if raw.empty:
                logger.warning("[feuille] %s: vide, skip", name)
                continue

            h = choose_header_row(raw, scan=25)
            cols = normalize_columns(raw.iloc[h].tolist())
            df = raw.iloc[h+1:].copy()
            df.columns = cols
            df = df.dropna(how="all")
            df = drop_empty_columns(df)
            if df.empty:
                logger.warning("[feuille] %s: vide après normalisation, skip", name)
                continue

            # inférence schéma
            schema = {}
            for c in df.columns:
                casted, tag = infer_col(df[c])
                df[c] = casted
                schema[c] = tag

            # table cible
            sheet_base = snake_id(name)
            table_name = pg_ident(file_base if len(sheets) == 1 else f"{file_base}__{sheet_base}")

            logger.info("Target: %s.%s | rows=%d cols=%d", TARGET_SCHEMA, table_name, len(df), df.shape[1])
            if SHOW_SAMPLE:
                logger.debug("Sample (top 8):\n%s", df.head(8).to_string(index=False))

            # ===== INSERTION DB (audit ici) =====
            try:
                df_sql = add_audit_columns(df.copy())
                dtype_map = {c: pg_type(schema[c]) for c in df.columns}
                dtype_map.update(audit_dtype_map())

                df_sql.to_sql(
                    name=table_name,
                    con=engine,
                    schema=TARGET_SCHEMA,
                    if_exists=IF_EXISTS_MODE,
                    index=False,
                    dtype=dtype_map,
                    method="multi",
                    chunksize=1000
                )
                logger.info("[ok] Inserted into %s.%s", TARGET_SCHEMA, table_name)
                done += 1
            except Exception:
                logger.exception("Échec insertion %s.%s", TARGET_SCHEMA, table_name)

        logger.info("Done. Feuilles analysées & importées: %d", done)

if __name__ == "__main__":
    main()
