"""Traitements autour des fichiers Excel."""

from __future__ import annotations

from typing import Iterable, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import re
import unicodedata
import warnings

from .db import sql_dtype

HEADER_HINTS = (
    "projet",
    "projets",
    "code",
    "id",
    "date",
    "mois",
    "année",
    "client",
    "fournisseur",
    "libelle",
    "libellé",
    "montant",
    "quantite",
    "qté",
    "qte",
    "prix",
    "ttc",
    "ht",
    "statut",
    "status",
    "site",
    "type",
    "categorie",
    "echeance",
    "échéance",
    "délai",
    "delai",
    "commentaire",
    "ref",
    "réf",
)

MAP_TRUE = {"true", "vrai", "oui", "yes", "1", "y", "o"}
MAP_FALSE = {"false", "faux", "non", "no", "0", "n"}

DATE_TOKEN_RE = re.compile(
    r"(?:\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})|(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janv|févr|fevr|avr|mai|juin|juil|sept|oct|nov|d[ée]c)",
    re.I,
)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?$")
COLNAME_DATE_HINTS = ("date", "dt", "heure", "time", "echeance", "échéance")


def strip_accents_lower(value) -> str:
    """Normalise les chaînes : supprime les accents et passe en minuscule."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower().strip()


def snake_id(value) -> str:
    """Crée un identifiant snake_case à partir d'une valeur quelconque."""

    text = strip_accents_lower(value)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\s\.-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "col"


def make_unique_columns(columns: Iterable[str]) -> List[str]:
    """Garantit l'unicité des noms de colonnes en ajoutant un suffixe numérique."""

    seen: dict[str, int] = {}
    result: List[str] = []
    for col in columns:
        base = col or "col"
        count = seen.get(base, 0)
        if count:
            new_name = f"{base}_{count+1}"
        else:
            new_name = base
        while new_name in seen:
            count += 1
            new_name = f"{base}_{count+1}"
        seen[base] = seen.get(base, 0) + 1
        seen[new_name] = 1
        result.append(new_name)
    return result


def _preclean_numeric_strings(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace("\u00A0", " ", regex=False)
        .str.replace("\u202F", " ", regex=False)
        .str.replace(" ", "", regex=False)
    )


def parse_number_like(series: pd.Series) -> pd.Series:
    cleaned = _preclean_numeric_strings(series)
    cleaned = cleaned.str.replace(r"(?<=\d)\.(?=\d{3}(?:\D|$))", "", regex=True)
    cleaned = cleaned.str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.apply(lambda col: col.dropna().astype(str).str.strip().ne("").any())
    return df.loc[:, mask]


def choose_header_row(df: pd.DataFrame, scan: int = 30) -> int:
    limit = min(len(df), scan)
    candidate, candidate_hits = None, -1
    for idx in range(limit):
        row = df.iloc[idx].astype(str)
        hits = sum(
            1 for value in row if value and any(hint in strip_accents_lower(value) for hint in HEADER_HINTS)
        )
        if hits >= 2 and hits > candidate_hits:
            candidate, candidate_hits = idx, hits
    if candidate is not None:
        return candidate

    best_density, best_idx = -1.0, 0
    for idx in range(limit):
        row = df.iloc[idx].astype(str)
        density = row.str.strip().ne("").mean()
        if density > best_density:
            best_density, best_idx = density, idx
    return best_idx


def normalize_columns(header_vals: Iterable) -> List[str]:
    raw = [snake_id(v) for v in header_vals]
    normalized = [name if name else f"col_{i+1}" for i, name in enumerate(raw)]
    return make_unique_columns(normalized)


def excel_serial_to_datetime(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    values[~np.isfinite(values)] = np.nan
    mask = (values >= 60.0) & (values <= 106750.0)
    values = values.where(mask, np.nan)
    base = pd.Timestamp("1899-12-30")
    delta = pd.to_timedelta(values, unit="D", errors="coerce")
    result = base + delta
    return pd.to_datetime(result, errors="coerce")


def infer_col(col: pd.Series) -> Tuple[pd.Series, str]:
    """Déduit le type d'une colonne et retourne la série castée et le tag."""

    series = col.copy()
    as_str = series.astype("string").str.strip()
    as_str = as_str.where(~as_str.fillna("").eq("0"), pd.NA)

    iso_mask = as_str.fillna("").str.match(ISO_DATE_RE)
    if iso_mask.mean() >= 0.5:
        dt_iso = pd.to_datetime(as_str.where(iso_mask), errors="coerce", dayfirst=False)
        if dt_iso.notna().mean() >= 0.5:
            has_time = (dt_iso.dt.time.astype(str) != "00:00:00").mean() > 0.2
            return dt_iso, "DATETIME" if has_time else "DATE"

    numeric = pd.to_numeric(as_str, errors="coerce")
    if len(as_str):
        safe_ratio = ((numeric >= 60) & (numeric <= 106750)).mean()
    else:
        safe_ratio = 0.0
    if numeric.notna().mean() >= 0.9 and safe_ratio >= 0.7:
        parsed = excel_serial_to_datetime(numeric)
        if parsed.notna().mean() >= 0.7:
            has_time = (parsed.dt.time.astype(str) != "00:00:00").mean() > 0.2
            return parsed, "DATETIME" if has_time else "DATE"

    colname_hint = any(hint in strip_accents_lower(series.name or "") for hint in COLNAME_DATE_HINTS)
    date_token_ratio = as_str.fillna("").str.contains(DATE_TOKEN_RE, na=False).mean()
    if colname_hint or date_token_ratio >= 0.30:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Could not infer format.*", category=UserWarning
            )
            dt1 = pd.to_datetime(as_str, errors="coerce", dayfirst=True, cache=True)
            dt2 = pd.to_datetime(as_str, errors="coerce", dayfirst=False, cache=True)
        dt = dt2 if dt2.notna().sum() > dt1.notna().sum() else dt1
        ok_ratio = dt.notna().mean()
        threshold = 0.50 if colname_hint else 0.70
        if ok_ratio >= threshold:
            has_time = (dt.dt.time.astype(str) != "00:00:00").mean() > 0.2
            return dt, "DATETIME" if has_time else "DATE"

    bool_series = as_str.str.lower().map(
        lambda value: True if value in MAP_TRUE else False if value in MAP_FALSE else pd.NA
    )
    if bool_series.notna().mean() >= 0.98:
        return bool_series.astype("boolean"), "BOOL"

    numbers = parse_number_like(as_str)
    if numbers.notna().mean() >= 0.85:
        non_zero = numbers.dropna()
        if len(non_zero) and (np.mod(non_zero, 1) == 0).all():
            return numbers.astype("Int64"), "INT"
        return numbers.astype("Float64"), "FLOAT"

    return as_str.astype("string"), "STRING"


def prepare_for_sql(df: pd.DataFrame, schema_tags: dict) -> Tuple[pd.DataFrame, dict]:
    output = df.copy()
    for column, tag in schema_tags.items():
        if tag == "DATE":
            output[column] = pd.to_datetime(output[column], errors="coerce").dt.date
        elif tag == "DATETIME":
            output[column] = pd.to_datetime(output[column], errors="coerce")
    dtype_map = {column: sql_dtype(tag) for column, tag in schema_tags.items()}
    return output, dtype_map


def resolve_table_name(file_stem: str, sheet_name: str | None, total_sheets: int) -> str:
    file_base = snake_id(file_stem)
    if total_sheets == 1 or not sheet_name:
        return file_base
    return f"{file_base}__{snake_id(sheet_name)}"


def load_excel(path: Path) -> dict[str, pd.DataFrame]:
    engine = "openpyxl"
    if path.suffix.lower() == ".xlsb":
        engine = "pyxlsb"
    return pd.read_excel(path, sheet_name=None, engine=engine, header=None)


def pg_ident(name: str, maxlen: int = 63) -> str:
    return name[:maxlen]
