"""Script très simple pour envoyer un classeur Excel dans PostgreSQL."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import create_engine


REQUIRED_ENV_VARS = ("PG_DB", "PG_USER", "PG_PASS")


@dataclass
class RuntimeOptions:
    """Options calculées à partir des arguments CLI."""

    source_path: Path
    table_name: str
    schema: str
    if_exists: str
    sheet_name: str | None


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments minimal."""

    parser = argparse.ArgumentParser(
        description="Charge un fichier Excel dans une table PostgreSQL"
    )
    parser.add_argument("source", help="Chemin vers le fichier .xlsx/.xls")
    parser.add_argument(
        "--table",
        help="Nom de la table cible (défaut: nom du fichier)",
    )
    parser.add_argument(
        "--schema",
        default="public",
        help="Schéma PostgreSQL cible (défaut: public)",
    )
    parser.add_argument(
        "--if-exists",
        choices=["fail", "replace", "append"],
        default="fail",
        help="Que faire si la table existe déjà",
    )
    parser.add_argument(
        "--sheet",
        help="Nom de la feuille à charger (défaut: toutes les feuilles)",
    )
    return parser


def normalise_name(value: str) -> str:
    """Nettoie un nom pour qu'il soit compatible avec PostgreSQL."""

    cleaned = value.strip().lower()
    cleaned = cleaned.replace(" ", "_")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned or "table"


def detect_table_name(path: Path, override: str | None) -> str:
    if override:
        return normalise_name(override)
    return normalise_name(path.stem)


def collect_options(args: argparse.Namespace) -> RuntimeOptions:
    source_path = Path(args.source).expanduser().resolve()
    if not source_path.exists():
        raise SystemExit(f"Fichier introuvable: {source_path}")

    table_name = detect_table_name(source_path, args.table)
    return RuntimeOptions(
        source_path=source_path,
        table_name=table_name,
        schema=args.schema,
        if_exists=args.if_exists,
        sheet_name=args.sheet,
    )


def get_database_url() -> str:
    """Construit l'URL SQLAlchemy à partir des variables d'environnement."""

    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        missing_vars = ", ".join(missing)
        raise SystemExit(
            "Variables d'environnement manquantes: "
            f"{missing_vars}. Complétez .env puis exécutez `export $(cat .env)`"
        )

    user = os.environ["PG_USER"]
    password = os.environ["PG_PASS"]
    database = os.environ["PG_DB"]
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def load_sheets(path: Path, sheet: str | None) -> Dict[str, pd.DataFrame]:
    """Retourne un dictionnaire {nom_feuille: dataframe}."""

    if sheet:
        print(f"Lecture de la feuille '{sheet}'...")
        return {sheet: pd.read_excel(path, sheet_name=sheet)}

    workbook = pd.ExcelFile(path)
    sheets: Dict[str, pd.DataFrame] = {}
    for sheet_name in workbook.sheet_names:
        print(f"Lecture de la feuille '{sheet_name}'...")
        sheets[sheet_name] = workbook.parse(sheet_name)
    return sheets


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes/colonnes vides et normalise les colonnes."""

    df = df.dropna(how="all")
    df = df.dropna(how="all", axis=1)
    df.columns = [normalise_name(str(col)) for col in df.columns]
    return df


def load_dataframe_to_postgres(
    engine_url: str,
    data: pd.DataFrame,
    table_name: str,
    schema: str,
    if_exists: str,
) -> None:
    """Envoie le dataframe dans PostgreSQL via pandas.to_sql."""

    engine = create_engine(engine_url, future=True)
    with engine.begin() as connection:
        data.to_sql(
            name=table_name,
            con=connection,
            schema=schema,
            if_exists=if_exists,
            index=False,
        )


def run(options: RuntimeOptions) -> None:
    print("--- Paramètres d'exécution ---")
    print(f"Fichier : {options.source_path}")
    print(f"Table  : {options.schema}.{options.table_name}")
    print(f"Mode   : {options.if_exists}")
    if options.sheet_name:
        print(f"Feuille: {options.sheet_name}")
    print("------------------------------")

    database_url = get_database_url()
    sheets = load_sheets(options.source_path, options.sheet_name)

    for sheet_name, raw_df in sheets.items():
        cleaned = clean_dataframe(raw_df)
        if cleaned.empty:
            print(f"[ignore] {sheet_name}: feuille vide après nettoyage")
            continue

        print(f"[upload] {sheet_name}: {len(cleaned)} lignes vers {options.table_name}")
        load_dataframe_to_postgres(
            engine_url=database_url,
            data=cleaned,
            table_name=options.table_name,
            schema=options.schema,
            if_exists=options.if_exists,
        )
        print(f"[ok] {sheet_name} importé")

    print("Terminé !")


def run_cli(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    options = collect_options(args)
    run(options)


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
