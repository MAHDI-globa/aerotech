# src/etl_fnp.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import numpy as np
import re, unicodedata, warnings

from dotenv import load_dotenv
from sqlalchemy import types as satypes, text
from sqlalchemy.exc import OperationalError

from .db import get_engine
from .audit import add_audit_columns, audit_dtype_map
from .logging_utils import get_logger, correlation  # ← LOGGING

# ============== ENV ==============
ROOT = Path(__file__).resolve().parent.parent
ENV = os.getenv("ENV", "dev").lower()

# fichiers: .env.files.{ENV} (fallback .env.files)
files_env = ROOT / f".env.files.{ENV}"
fallback_files_env = ROOT / ".env.files"
if files_env.exists():
    load_dotenv(files_env); files_env_loaded = files_env
elif fallback_files_env.exists():
    load_dotenv(fallback_files_env); files_env_loaded = fallback_files_env
else:
    raise RuntimeError(
        f"Env fichiers introuvable: {files_env} et {fallback_files_env}.\n"
        f"Ajoute p.ex.: FILE_PATH_FNP='C:/globasoft/aerotech/fic/FNP AEC projets - source.xlsx'"
    )

# BDD: .env.{ENV}
db_env = ROOT / f".env.{ENV}"
if not db_env.exists():
    raise RuntimeError(f"Env BDD introuvable: {db_env}")
load_dotenv(db_env)

FILE_PATH = os.getenv("FILE_PATH_FNP")
if not FILE_PATH:
    raise RuntimeError(f"FILE_PATH_FNP manquant dans {files_env_loaded}")

# ============== CONFIG ==============
SHOW_SAMPLE    = True
TARGET_SCHEMA  = "public"
IF_EXISTS_MODE = "replace"  # ou "append"

# ============== LOGGING ==============
logger = get_logger("etl_fnp", logfile="etl_fnp.log")

# ============== UTILS ==============
def strip_accents_lower(s):
    if s is None or (isinstance(s, float) and pd.isna(s)): return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()

def snake_id(s):
    s = strip_accents_lower(s)
    s = re.sub(r"[\s\.\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "col"

def parse_number_like(series: pd.Series) -> pd.Series:
    x = series.astype("string").str.replace("\u00A0"," ", regex=False).str.replace(" ","", regex=False)
    x = x.str.replace(r"(?<=\d)\.(?=\d{3}(?:\D|$))","", regex=True)
    x = x.str.replace(",",".", regex=False)
    return pd.to_numeric(x, errors="coerce")

# ============== HEADER PICK + NORMALIZE ==============
HEADER_HINTS = (
    "projet","projets","code","id","date","mois","année","client","fournisseur","libelle","libellé",
    "montant","quantite","qté","qte","prix","ttc","ht","statut","status","site","type","categorie",
    "echeance","échéance","délai","delai","commentaire","ref","réf"
)

def choose_header_row(df: pd.DataFrame, scan=25) -> int:
    limit = min(len(df), scan)
    cand_idx, cand_hits = None, -1
    for i in range(limit):
        row = df.iloc[i].astype(str)
        hits = 0
        for v in row:
            t = strip_accents_lower(v)
            if t and any(k in t for k in HEADER_HINTS):
                hits += 1
        if hits >= 2 and hits > cand_hits:
            cand_idx, cand_hits = i, hits
    if cand_idx is not None:
        return cand_idx
    best, idx = -1, 0
    for i in range(limit):
        row = df.iloc[i].astype(str)
        non_empty = row.map(lambda x: x.strip()!="").sum()
        texty = row.map(lambda x: bool(re.search(r"[A-Za-zÀ-ÿ]", x))).sum()
        score = non_empty*2 + texty
        if score > best: best, idx = score, i
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
    blank_cols = [c for c in df.columns if df[c].dropna().astype(str).str.strip().eq("").all()]
    return df.drop(columns=blank_cols) if blank_cols else df

# ============== INFÉRENCE TYPES ==============
MAP_TRUE  = {"true","vrai","oui","yes","1","y","o"}
MAP_FALSE = {"false","faux","non","no","0","n"}

import re, warnings
DATE_TOKEN_RE = re.compile(
    r"(?:\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})|(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janv|févr|fevr|avr|mai|juin|juil|sept|oct|nov|d[ée]c)",
    re.I
)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?$")
COLNAME_DATE_HINTS = ("date","dt","heure","time","echeance","échéance")

def excel_serial_to_datetime(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype("float64")
    vals[~np.isfinite(vals)] = np.nan
    mask = (vals >= 60) & (vals <= 106750)
    vals = vals.where(mask, np.nan)
    base = pd.Timestamp("1899-12-30")
    return pd.to_datetime(base + pd.to_timedelta(vals, unit="D", errors="coerce"), errors="coerce")

def infer_col(col: pd.Series):
    s = col.copy()
    ss = s.astype("string").str.strip()
    ss = ss.where(~ss.fillna("").eq("0"), pd.NA)

    iso_mask = ss.fillna("").str.match(ISO_DATE_RE)
    if iso_mask.mean() >= 0.5:
        dt_iso = pd.to_datetime(ss.where(iso_mask), errors="coerce", dayfirst=False)
        ok = dt_iso.notna().mean() if len(ss) else 0.0
        if ok >= 0.5:
            has_time = (dt_iso.dt.time.astype(str) != "00:00:00").mean() > 0.2
            return (dt_iso, "DATETIME" if has_time else "DATE")

    as_num = pd.to_numeric(ss, errors="coerce")
    safe_ratio = ((as_num >= 60) & (as_num <= 106750)).mean() if len(ss) else 0.0
    if (as_num.notna().mean() if len(ss) else 0.0) >= 0.9 and safe_ratio >= 0.7:
        parsed = excel_serial_to_datetime(as_num)
        ok = parsed.notna().mean() if len(ss) else 0.0
        if ok >= 0.7:
            has_time = (parsed.dt.time.astype(str) != "00:00:00").mean() > 0.2
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
        ok = dt.notna().mean() if len(ss) else 0.0
        if ok >= (0.50 if colname_hint else 0.70):
            has_time = (dt.dt.time.astype(str) != "00:00:00").mean() > 0.2
            return (dt, "DATETIME" if has_time else "DATE")

    mb = ss.str.lower().map(lambda x: True if x in MAP_TRUE else (False if x in MAP_FALSE else pd.NA))
    if (mb.notna().mean() if len(ss) else 0) >= 0.98:
        return mb.astype("boolean"), "BOOL"

    nums = parse_number_like(ss)
    if (nums.notna().mean() if len(ss.dropna()) else 0) >= 0.85:
        if len(nums.dropna()) and (np.mod(nums.dropna(), 1) == 0).all():
            return nums.astype("Int64"), "INT"
        return nums.astype("Float64"), "FLOAT"

    return ss.astype("string"), "STRING"

# map tags -> SQLAlchemy types
from sqlalchemy import types as satypes
def sa_type(tag: str):
    return {
        "DATE": satypes.Date(),
        "DATETIME": satypes.DateTime(),
        "INT": satypes.Integer(),
        "FLOAT": satypes.Float(precision=53),
        "BOOL": satypes.Boolean(),
        "STRING": satypes.Text(),
    }.get(tag, satypes.Text())

# ============== WRITE ==============
def write_df(engine, df: pd.DataFrame, table: str, schema_tags: dict):
    # conversions pandas finales pour cohérence
    for c, tag in schema_tags.items():
        if tag == "DATE":
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
        elif tag == "DATETIME":
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # >>> AUDIT : ajoute _loaded_at / _loaded_by
    df = add_audit_columns(df)
    dtype_map = {c: sa_type(tag) for c, tag in schema_tags.items()}
    dtype_map.update(audit_dtype_map())  # types des colonnes d’audit

    df.to_sql(
        name=table,
        con=engine,
        schema=TARGET_SCHEMA,
        if_exists=IF_EXISTS_MODE,
        index=False,
        dtype=dtype_map,
        method="multi",
        chunksize=1000,
    )

# ============== MAIN ==============
def main():
    with correlation("fnp-"):
        xlsx = Path(FILE_PATH)
        logger.info("Start etl_fnp | file=%s env=%s files-env=%s", xlsx, ENV, getattr(files_env_loaded, "name", "?"))
        if not xlsx.exists():
            logger.error("Fichier introuvable: %s", xlsx)
            return

        # Lecture
        try:
            sheets = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl", header=None)
            logger.info("Workbook lu: %d feuille(s)", len(sheets))
        except Exception:
            logger.exception("Lecture échouée")
            return

        # Connexion BDD
        try:
            eng = get_engine()
            with eng.connect() as c:
                ver = c.execute(text("select version();")).scalar()
                logger.info("DB connectée — %s | ENV=%s | files=%s", ver, ENV, getattr(files_env_loaded, "name", "?"))
        except (OperationalError, RuntimeError):
            logger.exception("Connexion PostgreSQL échouée")
            return

        base_table = snake_id(xlsx.stem)
        done = 0

        for name, raw in sheets.items():
            logger.info("[feuille] %s: shape initiale=%s", name, raw.shape)
            raw = raw.dropna(how="all", axis=0).dropna(how="all", axis=1)
            logger.info("[feuille] %s: après drop vides -> %s", name, raw.shape)
            if raw.empty:
                logger.warning("[feuille] %s: vide, on passe.", name)
                continue

            h = choose_header_row(raw, scan=25)
            cols = normalize_columns(raw.iloc[h].tolist())
            df = raw.iloc[h+1:].copy()
            df.columns = cols
            df = df.dropna(how="all")
            df = drop_empty_columns(df)
            logger.info("[feuille] %s: shape après normalisation -> %s", name, df.shape)
            if df.empty:
                logger.warning("[feuille] %s: vide après normalisation, on passe.", name)
                continue

            schema = {}
            for c in df.columns:
                casted, tag = infer_col(df[c])
                df[c] = casted
                schema[c] = tag

            table = base_table if len(sheets) == 1 else f"{base_table}__{snake_id(name)}"

            logger.info("Target table: %s.%s | rows=%d cols=%d", TARGET_SCHEMA, table, len(df), df.shape[1])
            if SHOW_SAMPLE:
                logger.debug("Sample (top 10):\n%s", df.head(10).to_string(index=False))

            try:
                write_df(eng, df.copy(), table, schema)
                logger.info("[ok] Inserted into %s.%s", TARGET_SCHEMA, table)
                done += 1
            except Exception:
                logger.exception("Échec insertion %s.%s", TARGET_SCHEMA, table)

        logger.info("Done. Feuilles analysées & importées: %d", done)

if __name__ == "__main__":
    main()
