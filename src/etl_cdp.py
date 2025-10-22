# src/etl_cdp.py
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import re, unicodedata

from sqlalchemy import text
from sqlalchemy import types as satypes
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

from .db import get_engine
from .audit import add_audit_columns, audit_dtype_map
from .logging_utils import get_logger, correlation  # ← LOGGING

# ========= CONFIG =========
SHOW_SAMPLE    = True
TARGET_SCHEMA  = "public"
IF_EXISTS_MODE = "replace"   # "replace" ou "append"
TABLE_NAME     = None        # None => snake(file_stem) ; sinon ex: "conditions_paiement"

# ========= LOGGING =========
logger = get_logger("etl_cdp", logfile="etl_cdp.log")

# ========= CHARGEMENT ENV =========
ROOT = Path(__file__).resolve().parent.parent
ENV = os.getenv("ENV", "dev").lower()

db_env_file = ROOT / f".env.{ENV}"
if not db_env_file.exists():
    raise RuntimeError(f"Fichier .env BDD introuvable: {db_env_file}")
load_dotenv(db_env_file)

files_env_file = ROOT / f".env.files.{ENV}"
fallback_files_env = ROOT / ".env.files"
if files_env_file.exists():
    load_dotenv(files_env_file)
    files_env_loaded = files_env_file
elif fallback_files_env.exists():
    load_dotenv(fallback_files_env)
    files_env_loaded = fallback_files_env
else:
    raise RuntimeError(
        f"Fichier d'env fichiers introuvable: {files_env_file} et {fallback_files_env}.\n"
        f"Crée l'un des deux avec, par ex.:\nFILE_PATH_CDP='C:/globasoft/aerotech/fic/Conditionsdepaiement.xlsx'"
    )

FILE_PATH = os.getenv("FILE_PATH_CDP")
if not FILE_PATH:
    raise RuntimeError(f"FILE_PATH_CDP manquant dans {files_env_loaded}")

# ========= UTILS =========
def strip_accents_lower(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()

def snake_id(s: str) -> str:
    s = strip_accents_lower(s)
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"[\s\.\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "col"

def parse_number_like(series: pd.Series) -> pd.Series:
    x = series.astype("string") \
        .str.replace("\u00A0", " ", regex=False) \
        .str.replace("\u202F", " ", regex=False) \
        .str.replace(" ", "", regex=False)
    x = x.str.replace(r"(?<=\d)\.(?=\d{3}(?:\D|$))", "", regex=True)
    x = x.str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")

def pick_numeric_dtype(s: pd.Series):
    if pd.api.types.is_integer_dtype(s):
        return satypes.Integer()
    if pd.api.types.is_float_dtype(s):
        return satypes.Float(precision=53)
    return satypes.Text()

# ========= ECRITURE DB =========
def dtype_map_for_postgres(df: pd.DataFrame) -> dict:
    dtypes = {}
    for c in df.columns:
        dtypes[c] = pick_numeric_dtype(df[c]) if pd.api.types.is_numeric_dtype(df[c]) else satypes.Text()
    return dtypes

def write_to_db(engine, df: pd.DataFrame, table_name: str):
    # === AUDIT: ajoute _loaded_at / _loaded_by
    df = add_audit_columns(df)
    dtypes = dtype_map_for_postgres(df)
    dtypes.update(audit_dtype_map())   # types des colonnes d’audit

    df.to_sql(
        name=table_name,
        con=engine,
        schema=TARGET_SCHEMA,
        if_exists=IF_EXISTS_MODE,
        index=False,
        dtype=dtypes,
        method="multi",
        chunksize=1000
    )

# ========= MAIN =========
def main():
    with correlation("cdp-"):
        xlsx = Path(FILE_PATH)
        logger.info("Start etl_cdp | file=%s env=%s files-env=%s", xlsx, ENV, getattr(files_env_loaded, "name", "?"))
        if not xlsx.exists():
            logger.error("Fichier introuvable: %s", xlsx)
            return

        # Lecture du fichier
        try:
            df0 = pd.read_excel(xlsx, sheet_name=0, header=None, engine="openpyxl")
            logger.info("Feuille[0] lue: shape=%s", df0.shape)
        except Exception:
            logger.exception("Lecture échouée")
            return

        # Nettoyage vides
        df = df0.dropna(how="all", axis=0).dropna(how="all", axis=1)
        logger.info("Après drop vides: %s -> %s", df0.shape, df.shape)
        if df.empty:
            logger.warning("Feuille vide après nettoyage.")
            return

        # Respect strict du schéma
        if df.shape[1] >= 3:
            df = df.iloc[:, :3].copy()
            df.columns = ["code_condition", "libelle", "delai_source"]
            logger.info("Colonnes conservées: code_condition, libelle, delai_source")
        elif df.shape[1] == 2:
            df = df.iloc[:, :2].copy()
            df.columns = ["code_condition", "libelle"]
            logger.info("Colonnes conservées: code_condition, libelle")
        else:
            logger.warning("Moins de 2 colonnes non vides trouvées. Abandon.")
            return

        # Nettoyages légers
        if "libelle" in df.columns:
            df["libelle"] = df["libelle"].astype("string").str.strip()

        if "code_condition" in df.columns:
            s = df["code_condition"].astype("string").str.strip()
            non_na = s.dropna()
            all_digits = non_na.str.fullmatch(r"\d+").all() if len(non_na) else True
            if all_digits:
                df["code_condition"] = parse_number_like(df["code_condition"]).astype("Int64")

        if "delai_source" in df.columns:
            parsed = parse_number_like(df["delai_source"])
            if parsed.dropna().apply(float.is_integer).all():
                df["delai_source"] = parsed.astype("Int64")

        logger.info("Aperçu: rows=%d cols=%d", len(df), df.shape[1])
        if SHOW_SAMPLE:
            logger.debug("Sample (top 15):\n%s", df.head(15).to_string(index=False))

        file_base = snake_id(xlsx.stem)
        table_name = TABLE_NAME or file_base

        # Connexion DB
        try:
            engine = get_engine()
            with engine.connect() as conn:
                ver = conn.execute(text("select version();")).scalar()
                logger.info("DB connectée — %s | ENV=%s | files=%s", ver, ENV, getattr(files_env_loaded, "name", "?"))
        except (OperationalError, RuntimeError):
            logger.exception("Connexion PostgreSQL échouée")
            return df

        # Écriture
        try:
            write_to_db(engine, df, table_name)
            logger.info("Inserted into %s.%s | rows=%d", TARGET_SCHEMA, table_name, len(df))
        except Exception:
            logger.exception("Échec insertion %s.%s", TARGET_SCHEMA, table_name)

        return df

if __name__ == "__main__":
    df_conditions = main()
