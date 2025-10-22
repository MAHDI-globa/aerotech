# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import numpy as np
import re, unicodedata, warnings

from sqlalchemy import text, types as satypes
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

from .db import get_engine
from .audit import add_audit_columns, audit_dtype_map
from .logging_utils import get_logger, correlation  # ← LOGGING

# ============== CHARGEMENT ENV ==============
ROOT = Path(__file__).resolve().parent.parent
ENV = os.getenv("ENV", "dev").lower()

# 1) BDD
db_env = ROOT / f".env.{ENV}"
if not db_env.exists():
    raise RuntimeError(f"Fichier .env BDD introuvable: {db_env}")
load_dotenv(db_env)

# 2) Fichiers (chemins)
files_env = ROOT / f".env.files.{ENV}"
fallback_files_env = ROOT / ".env.files"
if files_env.exists():
    load_dotenv(files_env); files_env_loaded = files_env
elif fallback_files_env.exists():
    load_dotenv(fallback_files_env); files_env_loaded = fallback_files_env
else:
    raise RuntimeError(
        f"Env fichiers introuvable: {files_env} et {fallback_files_env}.\n"
        f"Ajoute p.ex.:\nFILE_PATH_SPA='C:/globasoft/aerotech/fic/Suivi Projets AEC - source.xlsx'"
    )

FILE_PATH = os.getenv("FILE_PATH_SPA")
if not FILE_PATH:
    raise RuntimeError(f"FILE_PATH_SPA manquant dans {files_env_loaded}")

# ============== CONFIG ==============
SHOW_SAMPLE = True
TARGET_SCHEMA = "public"
IF_EXISTS_MODE = "replace"  # "replace" ou "append"
warnings.filterwarnings(
    "ignore",
    message=r"Parsing dates in %Y-%m-%d %H:%M:%S format.*",
    category=UserWarning
)

# ============== LOGGER ==============
logger = get_logger("etl_spa", logfile="etl_spa.log")

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

def make_unique_columns(cols):
    out, seen = [], {}
    for c in cols:
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    return out

def _preclean_numeric_strings(x: pd.Series) -> pd.Series:
    s = x.astype("string")
    s = s.str.replace(r"^(true|false|yes|no|vrai|faux)$", "", flags=re.I, regex=True)
    s = s.str.replace("\u00A0"," ", regex=False).str.replace(" ","", regex=False)
    s = s.str.replace(r"(?<=\d)\.(?=\d{3}(?:\D|$))","", regex=True)  # 1.234 -> 1234
    s = s.str.replace(",",".", regex=False)                          # 1,23 -> 1.23
    return s

def parse_number_like(series: pd.Series) -> pd.Series:
    return pd.to_numeric(_preclean_numeric_strings(series), errors="coerce")

def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    blank = [c for c in df.columns if df[c].dropna().astype(str).str.strip().eq("").all()]
    return df.drop(columns=blank) if blank else df

# ============== HEADER PICK ==============
HEADER_HINTS = (
    "projet","projets","code","id","réf","ref","libelle","libellé","intitulé",
    "client","fournisseur","chef de projet","cp","cdp","manager",
    "statut","status","phase","categorie","catégorie","type",
    "date","heure","time","période","periode","echeance","échéance","fin","début","debut",
    "budget","montant","cout","coût","prix","ht","ttc","avancement","progress","%","taux",
    "site","section analytique","analytique","commentaire","observations","notes","remarques",
    "comments","revue raf","statut clipper","gtm","pôle","pole",
    "remaining forecast","facturation mois","hrs","heures","(h)",
    "ca ","charges","marge","achats","commandé","enregistré","fnp","int","raf"
)

def choose_header_row(df: pd.DataFrame, scan=30) -> int:
    limit = min(len(df), scan)
    cand_idx, cand_hits = None, -1
    for i in range(limit):
        row = df.iloc[i].astype(str)
        hits = sum(1 for v in row if (t:=strip_accents_lower(v)) and any(k in t for k in HEADER_HINTS))
        if hits >= 2 and hits > cand_hits:
            cand_idx, cand_hits = i, hits
    if cand_idx is not None: return cand_idx
    best, idx = -1, 0
    for i in range(limit):
        row = df.iloc[i].astype(str)
        non_empty = row.map(lambda x: x.strip()!="").sum()
        texty = row.map(lambda x: bool(re.search(r"[A-Za-zÀ-ÿ]", x))).sum()
        score = non_empty*2 + texty
        if score > best: best, idx = score, i
    return idx

# ============== INFÉRENCE TYPES ==============
MAP_TRUE  = {"true","vrai","oui","yes","1","y","o"}
MAP_FALSE = {"false","faux","non","no","0","n"}

DATE_TOKEN_RE = re.compile(
    r"(?:\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})|(?:\d{1,2}[-/\.]\d{1,2][-/.]\d{2,4})|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janv|févr|fevr|avr|mai|juin|juil|sept|oct|nov|d[ée]c)",
    re.I
)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?$")

DATE_NAME_HINTS   = ("date","dt","heure","time","echeance","échéance","debut","début","fin","période","periode","semaine","mois","année","annee")
MONEY_NAME_HINTS  = ("€","ht","budget","montant","amount","total","prix","coût","cout","facturation","forecast","achats","commandé","enregistré","fnp","ca ","charges","marge","int","raf")
HOUR_NAME_HINTS   = ("heures","hrs","(h)"," h ")
PERCENT_NAME_HINTS= ("%", "taux", "marge", "avancement", "écart", "ecart", "pourcent")
CODE_LIKE_HINTS   = ("code","nature","plan","compte","sop","section","niveau","trigramme","immat","ot","ata","wp","n° dossier","n_dossier","dossier","cat point","cat pointage","statut clipper","gtm","pôle","pole","comments","cdp","top custo","réunion raf","reunion raf","statut suivi de projets","statut suivi de projet")

def colname_has(hints, name):
    n = strip_accents_lower(name or "")
    return any(h in n for h in hints)

def excel_serial_to_datetime(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype("float64")
    vals[~np.isfinite(vals)] = np.nan
    mask = (vals >= 60) & (vals <= 60000)
    base = pd.Timestamp("1899-12-30")
    return pd.to_datetime(
        base + pd.to_timedelta(vals.where(mask, np.nan), unit="D", errors="coerce"),
        errors="coerce"
    )

def infer_col(col: pd.Series):
    """Retourne (serie_casted, tag) ∈ {"DATE","DATETIME","INT","FLOAT","BOOL","STRING"}."""
    s = col.copy()
    ss = s.astype("string").str.strip()
    name = s.name or ""

    # argent / métriques -> float
    if colname_has(MONEY_NAME_HINTS, name):
        nums = parse_number_like(ss)
        if nums.notna().any(): return nums.astype("Float64"), "FLOAT"

    # heures -> float
    if colname_has(HOUR_NAME_HINTS, name):
        nums = parse_number_like(ss)
        if nums.notna().any(): return nums.astype("Float64"), "FLOAT"

    # pourcentage / taux -> float
    if ("%" in name) or colname_has(PERCENT_NAME_HINTS, name):
        nums = parse_number_like(ss)
        if nums.notna().any(): return nums.astype("Float64"), "FLOAT"

    # dates ISO
    iso_mask = ss.fillna("").str.match(ISO_DATE_RE)
    if len(ss) and iso_mask.mean() >= 0.5:
        dt_iso = pd.to_datetime(ss.where(iso_mask), errors="coerce", dayfirst=False)
        if dt_iso.notna().mean() >= 0.5:
            has_time = (dt_iso.dt.time.astype(str) != "00:00:00").mean() > 0.2
            return (dt_iso, "DATETIME" if has_time else "DATE")

    # Excel serial (si nom "datey")
    if colname_has(DATE_NAME_HINTS, name):
        as_num = pd.to_numeric(ss, errors="coerce")
        safe_mask = (as_num >= 60) & (as_num <= 60000)
        if (as_num.notna().mean() if len(ss) else 0.0) >= 0.9 and safe_mask.mean() >= 0.7:
            parsed = excel_serial_to_datetime(as_num)
            if parsed.notna().mean() >= 0.7:
                has_time = (parsed.dt.time.astype(str) != "00:00:00").mean() > 0.2
                return (parsed, "DATETIME" if has_time else "DATE")

    # tokens de date ou nom "datey"
    date_token_ratio = ss.fillna("").str.contains(DATE_TOKEN_RE, na=False).mean()
    if colname_has(DATE_NAME_HINTS, name) or (date_token_ratio >= 0.30):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)
            dt1 = pd.to_datetime(ss, errors="coerce", dayfirst=True,  cache=True)
            dt2 = pd.to_datetime(ss, errors="coerce", dayfirst=False, cache=True)
        dt = dt2 if dt2.notna().sum() > dt1.notna().sum() else dt1
        ok = dt.notna().mean() if len(ss) else 0.0
        if ok >= (0.50 if colname_has(DATE_NAME_HINTS, name) else 0.70):
            has_time = (dt.dt.time.astype(str) != "00:00:00").mean() > 0.2
            return (dt, "DATETIME" if has_time else "DATE")

    # bool strict
    mb = ss.str.lower().map(lambda x: True if x in MAP_TRUE else (False if x in MAP_FALSE else pd.NA))
    if len(ss) and mb.notna().mean() >= 0.98:
        return mb.astype("boolean"), "BOOL"

    # nombre générique
    nums = parse_number_like(ss)
    if (nums.notna().mean() if len(ss) else 0) >= 0.85:
        nz = nums.dropna()
        if len(nz) and (np.mod(nz, 1) == 0).all(): return nums.astype("Int64"), "INT"
        return nums.astype("Float64"), "FLOAT"

    return ss.astype("string"), "STRING"

def sa_type(tag: str):
    return {
        "DATE": satypes.Date(),
        "DATETIME": satypes.DateTime(),
        "INT": satypes.Integer(),
        "FLOAT": satypes.Float(precision=53),
        "BOOL": satypes.Boolean(),
        "STRING": satypes.Text()
    }.get(tag, satypes.Text())

def prepare_for_sql(df: pd.DataFrame, schema_tags: dict):
    out = df.copy()
    for c, tag in schema_tags.items():
        if tag == "DATE": out[c] = pd.to_datetime(out[c], errors="coerce").dt.date
        elif tag == "DATETIME": out[c] = pd.to_datetime(out[c], errors="coerce")
    dtype_map = {c: sa_type(tag) for c, tag in schema_tags.items()}
    return out, dtype_map

def pg_ident(name: str, maxlen=63) -> str:
    return name[:maxlen]

# ============== MAIN ==============
def main():
    with correlation("spa-"):
        xlsx = Path(FILE_PATH)
        logger.info("Start etl_spa | file=%s env=%s files-env=%s", xlsx, ENV, getattr(files_env_loaded, "name", "?"))
        if not xlsx.exists():
            logger.error("Fichier introuvable: %s", xlsx)
            return

        # Connexion BDD
        try:
            engine = get_engine()
            with engine.connect() as conn:
                ver = conn.execute(text("select version();")).scalar()
                logger.info("DB connectée — %s | ENV=%s | files=%s", ver, ENV, getattr(files_env_loaded, "name", "?"))
        except (OperationalError, RuntimeError):
            logger.exception("Connexion PostgreSQL échouée")
            return

        # Lecture multi-feuilles
        try:
            sheets = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl", header=None)
            logger.info("Workbook lu: %d feuille(s)", len(sheets))
        except Exception:
            logger.exception("Lecture Excel échouée")
            return

        file_base = snake_id(xlsx.stem)
        done = 0

        for name, raw in sheets.items():
            logger.info("[feuille] %s: shape initiale=%s", name, raw.shape)

            raw = raw.dropna(how="all", axis=0).dropna(how="all", axis=1)
            logger.debug("[feuille] %s: après drop vides -> %s", name, raw.shape)
            if raw.empty:
                logger.warning("[feuille] %s: vide, skip", name)
                continue

            h = choose_header_row(raw, scan=30)
            logger.debug("[feuille] %s: header choisi à la ligne %d", name, h)

            # Schéma STRICT du fichier
            df_src = raw.iloc[h+1:].copy()
            src_cols_original = raw.iloc[h].tolist()
            df_src.columns = src_cols_original
            df_src = df_src.dropna(how="all").copy()
            df_src = drop_empty_columns(df_src)
            logger.debug("[feuille] %s: shape après en-têtes -> %s", name, df_src.shape)
            if df_src.empty:
                logger.warning("[feuille] %s: vide après normalisation, skip", name)
                continue

            # noms Postgres
            pg_cols = make_unique_columns([snake_id(c) for c in df_src.columns])

            # inférence -> vrais dtypes pandas
            schema_src, casted_df = {}, df_src.copy()
            for c in list(df_src.columns):
                casted, tag = infer_col(df_src[c])
                casted_df[c] = casted
                schema_src[c] = tag

            table_base = file_base if len(sheets) == 1 else f"{file_base}__{snake_id(name)}"
            table_name = pg_ident(table_base)

            logger.info("Target: %s.%s | rows=%d cols=%d", TARGET_SCHEMA, table_name, len(casted_df), casted_df.shape[1])
            if SHOW_SAMPLE:
                try:
                    logger.debug("Sample (top 10):\n%s", casted_df.head(10).to_string(index=False))
                except Exception:
                    pass

            # ===== INSERTION DB =====
            try:
                df_for_sql = casted_df.copy()
                df_for_sql.columns = pg_cols
                schema_sql = {pg: schema_src[src] for src, pg in zip(df_src.columns, pg_cols)}
                df_sql, dtype_map = prepare_for_sql(df_for_sql, schema_sql)

                # audit
                df_sql = add_audit_columns(df_sql)
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
