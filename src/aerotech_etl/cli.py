"""Mini script d'import Excel → PostgreSQL."""

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Envoie une ou toutes les feuilles d'un classeur Excel vers PostgreSQL.",
        prog="aerotech-extract",
    )
    parser.add_argument("fichier", help="Chemin du classeur .xlsx/.xls")
    parser.add_argument(
        "--table",
        help="Nom de la table cible (défaut: nom du fichier en minuscules)",
    )
    parser.add_argument(
        "--schema",
        default="public",
        help="Schéma PostgreSQL à utiliser (défaut: public)",
    )
    parser.add_argument(
        "--if-exists",
        choices=["fail", "replace", "append"],
        default="fail",
        help="Comportement si la table existe déjà",
    )
    parser.add_argument(
        "--sheet",
        help="Nom d'une feuille précise (sinon toutes les feuilles sont importées)",
    )
    return parser.parse_args(argv)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(
            f"La variable d'environnement {name} est obligatoire. "
            "Remplissez .env puis exécutez `export $(grep -v '^#' .env | xargs)`"
        )
    return value


def build_engine_url() -> str:
    user = require_env("PG_USER")
    password = require_env("PG_PASS")
    database = require_env("PG_DB")
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def clean_columns(columns: list[str]) -> list[str]:
    cleaned = []
    for column in columns:
        name = str(column).strip().lower().replace(" ", "_")
        cleaned.append(name or "colonne")
    return cleaned


def import_sheet(path: Path, sheet: str | int | None, table: str, schema: str, if_exists: str) -> None:
    if sheet is None:
        sheets = pd.ExcelFile(path).sheet_names
    else:
        sheets = [sheet]

    engine_url = build_engine_url()
    engine = create_engine(engine_url, future=True)

    for sheet_name in sheets:
        print(f"→ Lecture de '{sheet_name}'")
        frame = pd.read_excel(path, sheet_name=sheet_name)
        frame = frame.dropna(how="all").dropna(how="all", axis=1)
        frame.columns = clean_columns(list(frame.columns))

        if frame.empty:
            print(f"  (aucune donnée, feuille ignorée)")
            continue

        print(f"  {len(frame)} lignes envoyées vers {schema}.{table}")
        frame.to_sql(
            name=table,
            con=engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.fichier).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Fichier introuvable: {source}")

    table = args.table or source.stem.lower().replace(" ", "_")
    import_sheet(source, args.sheet, table, args.schema, args.if_exists)
    print("Import terminé ✅")


if __name__ == "__main__":  # pragma: no cover - exécution directe
    main()
