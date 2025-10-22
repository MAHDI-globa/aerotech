# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import os, re, unicodedata
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text, types as satypes
from sqlalchemy.exc import OperationalError

from .db import get_engine
from .audit import add_audit_columns, audit_dtype_map
from .logging_utils import get_logger, correlation  # ← LOGGING

# ============== ENV & CONFIG ==============
ROOT = Path(__file__).resolve().parent.parent
ENV = os.getenv("ENV", "dev").lower()

# BDD
db_env = ROOT / f".env.{ENV}"
if not db_env.exists():
    raise RuntimeError(f"Fichier .env BDD introuvable: {db_env}")
load_dotenv(db_env)

# FICHIERS (avec fallback)
files_env = ROOT / f".env.files.{ENV}"
fallback_files_env = ROOT / ".env.files"
files_env_loaded = None
if files_env.exists():
    load_dotenv(files_env); files_env_loaded = files_env
elif fallback_files_env.exists():
    load_dotenv(fallback_files_env); files_env_loaded = fallback_files_env
else:
    raise RuntimeError(f"Env fichiers introuvable: {files_env} et {fallback_files_env}")

FILE_PATH = os.getenv("FILE_PATH_CATCLIENTS")
if not FILE_PATH:
    raise RuntimeError(f"FILE_PATH_CATCLIENTS manquant dans {files_env_loaded}")

TARGET_SCHEMA  = "public"
TARGET_TABLE   = "dim_client_categories"
IF_EXISTS_MODE = "replace"  # ou "append"
SHOW_SAMPLE    = True

# ============== LOGGING ==============
logger = get_logger("etl_catclients", logfile="etl_catclients.log")

# ============== UTILS ==============
def strip_accents_lower(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()

def snake_id(s: str) -> str:
    s = strip_accents_lower(s)
    s = re.sub(r"[\s\.\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "col"

HEADER_HINTS = ("libelle","libellé","libelle","top","custo","hfm","categorie","catégorie","category")

def choose_header_row(df: pd.DataFrame, scan: int = 25) -> int:
    limit = min(len(df), scan)
    best_idx, best_hits = 0, -1
    for i in range(limit):
        row = df.iloc[i].astype(str)
        hits = sum(1 for v in row if (t:=strip_accents_lower(v)) and any(k in t for k in HEADER_HINTS))
        if hits > best_hits:
            best_idx, best_hits = i, hits
    return best_idx if best_hits >= 1 else 0

CANON_MAP = {
    "libelle": "libelle",
    "libellé": "libelle",
    "libelle_": "libelle",
    "top_custo_hfm": "top_custo_hfm",
    "top_custo": "top_custo_hfm",
    "top": "top_custo_hfm",
    "categorie": "categorie",
    "catégorie": "categorie",
    "category": "categorie",
}

def canon_name(name: str) -> str:
    n = snake_id(name or "")
    if "top" in n and "hfm" in n:
        return "top_custo_hfm"
    return CANON_MAP.get(n, n)

def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    blank = [c for c in df.columns if df[c].astype(str).str.strip().replace("nan","").eq("").all()]
    return df.drop(columns=blank) if blank else df

def clean_text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace("\u00A0"," ", regex=False).str.strip()

# ============== MAIN ==============
def main():
    with correlation("catclients-"):
        xlsx = Path(FILE_PATH)
        logger.info("Start etl_catclients | file=%s env=%s files-env=%s",
                    xlsx, ENV, getattr(files_env_loaded, "name", "?"))

        if not xlsx.exists():
            logger.error("Fichier introuvable: %s", xlsx)
            return

        # Connexion DB
        try:
            engine = get_engine()
            with engine.connect() as c:
                ver = c.execute(text("select version();")).scalar()
                logger.info("DB connecté | %s", ver)
        except (OperationalError, RuntimeError) as e:
            logger.exception("Connexion PostgreSQL échouée")
            return

        # Lecture (toutes feuilles → concat)
        try:
            sheets = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl", header=None)
            logger.info("Fichier lu: %d feuille(s)", len(sheets))
        except Exception:
            logger.exception("Lecture échouée")
            return

        frames = []
        for name, raw in sheets.items():
            try:
                logger.debug("Feuille '%s' shape init=%s", name, raw.shape)
                raw = raw.dropna(how="all", axis=0).dropna(how="all", axis=1)
                if raw.empty:
                    logger.debug("Feuille '%s' vide après nettoyage", name)
                    continue
                h = choose_header_row(raw)
                df = raw.iloc[h+1:].copy()
                df.columns = [canon_name(c) for c in raw.iloc[h]]
                df = drop_empty_columns(df)
                if df.empty:
                    logger.debug("Feuille '%s' vide après normalisation", name)
                    continue
                frames.append(df)
            except Exception:
                logger.exception("Erreur traitement feuille '%s'", name)

        if not frames:
            logger.warning("Aucune donnée exploitable")
            return

        df = pd.concat(frames, ignore_index=True)

        # Ne garder que les 3 colonnes cibles si présentes
        keep = [c for c in ("libelle","top_custo_hfm","categorie") if c in df.columns]
        if not keep:
            logger.error("Colonnes attendues absentes (libelle/top_custo_hfm/categorie)")
            return
        df = df[keep].copy()

        # Nettoyage simple
        for c in keep:
            df[c] = clean_text(df[c])

        # Déduplication par libelle (priorité aux catégories non vides)
        if "libelle" in df.columns and "categorie" in df.columns:
            df["_nonempty_cat"] = df["categorie"].str.len().fillna(0) > 0
            df = df.sort_values(by=["_nonempty_cat"], ascending=False).drop(columns=["_nonempty_cat"])
        df = df.drop_duplicates(subset=["libelle"], keep="first")

        logger.info("Aperçu: rows=%d cols=%d", len(df), df.shape[1])
        if SHOW_SAMPLE:
            # on logge l’aperçu en DEBUG pour ne pas polluer en prod
            logger.debug("Sample (top 20):\n%s", df.head(20).to_string(index=False))

        # Écriture DB (TEXT + colonnes d'audit)
        dtype_map = {c: satypes.Text() for c in df.columns}
        df_sql = add_audit_columns(df)           # ← ajoute _loaded_at / _loaded_by
        dtype_map.update(audit_dtype_map())      # ← dtypes pour les colonnes d’audit

        try:
            df_sql.to_sql(
                name=TARGET_TABLE,
                con=engine,
                schema=TARGET_SCHEMA,
                if_exists=IF_EXISTS_MODE,
                index=False,
                dtype=dtype_map,
                method="multi",
                chunksize=1000,
            )
            logger.info("Inserted into %s.%s | rows=%d", TARGET_SCHEMA, TARGET_TABLE, len(df_sql))
        except Exception:
            logger.exception("Échec insertion %s.%s", TARGET_SCHEMA, TARGET_TABLE)

if __name__ == "__main__":
    main()
