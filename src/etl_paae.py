# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import os, re, unicodedata, warnings
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text, types as satypes
from sqlalchemy.exc import OperationalError

from .db import get_engine
from .audit import add_audit_columns, audit_dtype_map
from .logging_utils import get_logger, correlation  # ← LOGGING

# ==================== ENV & CONFIG ====================
ROOT = Path(__file__).resolve().parent.parent
ENV = os.getenv("ENV", "dev").lower()

# charge BDD
db_env = ROOT / f".env.{ENV}"
if not db_env.exists():
    raise RuntimeError(f"Fichier .env BDD introuvable: {db_env}")
load_dotenv(db_env)

# charge chemins fichiers
files_env = ROOT / f".env.files.{ENV}"
fallback_files_env = ROOT / ".env.files"
if files_env.exists():
    load_dotenv(files_env); files_env_loaded = files_env
elif fallback_files_env.exists():
    load_dotenv(fallback_files_env); files_env_loaded = fallback_files_env
else:
    raise RuntimeError(f"Env fichiers introuvable: {files_env} et {fallback_files_env}")

# ⚠️ clé simple et cohérente
FILE_PATH = os.getenv("FILE_PATH_PAAE")
if not FILE_PATH:
    raise RuntimeError(f"FILE_PATH_PAAE manquant dans {files_env_loaded}")

TARGET_SCHEMA  = "public"
IF_EXISTS_MODE = "replace"  # ou "append"
SHOW_SAMPLE    = True
warnings.filterwarnings("ignore", category=UserWarning)

# ==================== LOGGER ====================
logger = get_logger("etl_paae", logfile="etl_paae.log")

# ==================== TEXT/NAMES UTILS ====================
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

def make_unique_columns(cols):
    out, seen = [], {}
    for c in cols:
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    return out

# ==================== SIMPLE CLEANERS ====================
def parse_number_like(series: pd.Series) -> pd.Series:
    """Nettoyage FR/US basique + € + espaces insécables, puis to_numeric."""
    s = series.astype("string")
    s = s.str.replace("\u00A0", " ", regex=False).str.replace("\u202F", " ", regex=False)
    s = s.str.replace("€", "", regex=False).str.replace(" ", "", regex=False)
    s = s.str.replace(r"(?<=\d)\.(?=\d{3}(?:\D|$))", "", regex=True)  # 1.234.567 -> 1234567
    s = s.str.replace(",", ".", regex=False)                          # 1,23 -> 1.23
    return pd.to_numeric(s, errors="coerce")

def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    blank = [c for c in df.columns if df[c].dropna().astype(str).str.strip().eq("").all()]
    return df.drop(columns=blank) if blank else df

# ==================== HEADER PICK (très simple) ====================
HEADER_HINTS = (
    "projet","code","id","réf","ref","libelle","libellé","intitulé","description",
    "client","fournisseur","cdp","chef de projet","manager","statut","status","phase",
    "categorie","catégorie","type","site","pole","pôle","section analytique",
    "date","heure","time","période","periode","début","debut","fin","echeance","échéance",
    "mois","année","annee","semaine","budget","montant","coût","cout","prix","ht","ttc",
    "charges","achats","marge","ca","fnp","int","raf","heures","hrs","(h)","taux","%","forecast","facturation"
)

def choose_header_row(df: pd.DataFrame, scan=30) -> int:
    limit = min(len(df), scan)
    # 1) ligne avec le plus de mots-clés
    best_idx, best_hits = 0, -1
    for i in range(limit):
        row = df.iloc[i].astype(str)
        hits = sum(1 for v in row if (t:=strip_accents_lower(v)) and any(k in t for k in HEADER_HINTS))
        if hits > best_hits:
            best_idx, best_hits = i, hits
    if best_hits >= 2:
        return best_idx
    # 2) fallback : densité de texte
    scores = []
    for i in range(limit):
        row = df.iloc[i].astype(str)
        non_empty = row.map(lambda x: x.strip()!="").sum()
        texty = row.map(lambda x: bool(re.search(r"[A-Za-zÀ-ÿ]", x))).sum()
        scores.append((non_empty*2 + texty, i))
    return max(scores)[1] if scores else 0

# ==================== TYPE INFERENCE (ultra light) ====================
MAP_TRUE  = {"true","vrai","oui","yes","1","y","o"}
MAP_FALSE = {"false","faux","non","no","0","n"}

DATE_TOKEN_RE = re.compile(
    r"(?:\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})|(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janv|févr|fevr|avr|mai|juin|juil|sept|oct|nov|d[ée]c)",
    re.I,
)

DATE_NAME_HINTS   = ("date","dt","heure","time","echeance","échéance","debut","début","fin","période","periode","semaine","mois","année","annee")
MONEY_NAME_HINTS  = ("montant","debit","débit","credit","crédit","solde","budget","ht","ttc","charges","achats","marge","ca","fnp","forecast","facturation")
FORCE_TEXT_NAMES  = ("nature","section_analytique","sop","plan_1_niveau_1","plan_3_niveau_1","plan4","plan5")

def infer_col(col: pd.Series):
    """Heuristique courte : priorise argent/temps/dates ; force certains codes en texte."""
    name_snake = snake_id(col.name or "")
    s = col.astype("string").str.strip()

    if any(k in name_snake for k in FORCE_TEXT_NAMES):
        return s.astype("string"), "STRING"

    if any(k in name_snake for k in MONEY_NAME_HINTS) or "%" in name_snake or "h" == name_snake or "(h)" in name_snake:
        nums = parse_number_like(s)
        if nums.notna().any():
            return nums.astype("Float64"), "FLOAT"

    looks_date = any(k in name_snake for k in DATE_NAME_HINTS) or s.fillna("").str.contains(DATE_TOKEN_RE, na=False).mean() >= 0.30
    if looks_date:
        dt1 = pd.to_datetime(s, errors="coerce", dayfirst=True)
        dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False)
        dt  = dt2 if dt2.notna().sum() > dt1.notna().sum() else dt1
        if dt.notna().mean() >= 0.50:
            has_time = (dt.dt.time.astype(str) != "00:00:00").mean() > 0.2
            return (dt, "DATETIME" if has_time else "DATE")

    mb = s.str.lower().map(lambda x: True if x in MAP_TRUE else (False if x in MAP_FALSE else pd.NA))
    if mb.notna().mean() >= 0.98:
        return mb.astype("boolean"), "BOOL"

    nums = parse_number_like(s)
    if nums.notna().mean() >= 0.85:
        nz = nums.dropna()
        if len(nz) and np.isclose(nz, np.round(nz)).all():
            return nums.round().astype("Int64"), "INT"      # BIGINT côté SQL
        return nums.astype("Float64"), "FLOAT"

    return s.astype("string"), "STRING"

def sa_type(tag: str):
    return {
        "DATE":     satypes.Date(),
        "DATETIME": satypes.DateTime(),
        "INT":      satypes.BigInteger(),   # ← plus sûr
        "FLOAT":    satypes.Float(precision=53),
        "BOOL":     satypes.Boolean(),
        "STRING":   satypes.Text(),
    }.get(tag, satypes.Text())

def prepare_for_sql(df: pd.DataFrame, schema_tags: dict):
    out = df.copy()
    for c, tag in schema_tags.items():
        if tag == "DATE":
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.date
        elif tag == "DATETIME":
            out[c] = pd.to_datetime(out[c], errors="coerce")
    dtype_map = {c: sa_type(tag) for c, tag in schema_tags.items()}
    return out, dtype_map

def pg_ident(name: str, maxlen=63) -> str:
    return name[:maxlen]

# ==================== MAIN ====================
def main():
    with correlation("paae-"):
        xlsx = Path(FILE_PATH)
        logger.info("Start etl_paae | file=%s env=%s files-env=%s", xlsx, ENV, getattr(files_env_loaded, "name", "?"))
        if not xlsx.exists():
            logger.error("Fichier introuvable: %s", xlsx)
            return

        # DB
        try:
            engine = get_engine()
            with engine.connect() as c:
                ver = c.execute(text("select version();")).scalar()
                logger.info("DB connectée — %s | ENV=%s | files=%s", ver, ENV, getattr(files_env_loaded, "name", "?"))
        except (OperationalError, RuntimeError):
            logger.exception("Connexion PostgreSQL échouée")
            return

        # lecture toutes feuilles
        try:
            sheets = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl", header=None)
            logger.info("Workbook lu: %d feuille(s)", len(sheets))
        except Exception:
            logger.exception("Lecture Excel échouée")
            return

        base = snake_id(xlsx.stem)
        done = 0

        for sheet_name, raw in sheets.items():
            logger.info("[feuille] %s: shape=%s", sheet_name, raw.shape)
            raw = raw.dropna(how="all", axis=0).dropna(how="all", axis=1)
            if raw.empty:
                logger.warning("[feuille] %s: vide, skip", sheet_name)
                continue

            h = choose_header_row(raw, scan=30)
            cols_src = list(raw.iloc[h])
            df = raw.iloc[h+1:].copy()
            df.columns = cols_src
            df = df.dropna(how="all")
            df = drop_empty_columns(df)
            if df.empty:
                logger.warning("[feuille] %s: vide après normalisation, skip", sheet_name)
                continue

            # types & renommage
            schema, casted = {}, df.copy()
            for c in df.columns:
                casted[c], schema[c] = infer_col(df[c])

            pg_cols = make_unique_columns([snake_id(c) for c in df.columns])
            casted.columns = pg_cols

            table = base if len(sheets) == 1 else f"{base}__{snake_id(sheet_name)}"
            table = pg_ident(table)

            logger.info("Target: %s.%s | rows=%d cols=%d", TARGET_SCHEMA, table, len(casted), casted.shape[1])
            if SHOW_SAMPLE:
                logger.debug("Sample (top 10):\n%s", casted.head(10).to_string(index=False))

            # écriture (ajout des colonnes d'audit juste avant to_sql)
            try:
                df_sql, dtype_map = prepare_for_sql(casted, {pg: schema[src] for src, pg in zip(df.columns, pg_cols)})

                # 🔹 audit
                df_sql = add_audit_columns(df_sql)
                dtype_map.update(audit_dtype_map())

                df_sql.to_sql(
                    name=table, con=engine, schema=TARGET_SCHEMA,
                    if_exists=IF_EXISTS_MODE, index=False, dtype=dtype_map,
                    method="multi", chunksize=1000
                )
                logger.info("[ok] Inserted into %s.%s", TARGET_SCHEMA, table)
                done += 1
            except Exception:
                logger.exception("Échec insertion %s.%s", TARGET_SCHEMA, table)

        logger.info("Done. Feuilles importées: %d", done)

if __name__ == "__main__":
    main()
