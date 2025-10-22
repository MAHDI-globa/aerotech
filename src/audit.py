# src/audit.py
import os, getpass, pandas as pd
from sqlalchemy import types as satypes

AUDIT_COL_AT = "created_at"
AUDIT_COL_BY = "created_by"

def resolve_loader_user() -> str:
    return (
        os.getenv("LOADER_USER")
        or os.getenv("USERNAME")
        or os.getenv("USER")
        or getpass.getuser()
        or "unknown"
    )

def add_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[AUDIT_COL_AT] = pd.Timestamp.now(tz="UTC")    # horodatage UTC
    out[AUDIT_COL_BY] = resolve_loader_user()         # opérateur
    return out

def audit_dtype_map() -> dict:
    return {
        AUDIT_COL_AT: satypes.DateTime(timezone=True),  # timestamptz
        AUDIT_COL_BY: satypes.Text(),
    }
