"""Interface en ligne de commande pour lancer l'extraction."""

from __future__ import annotations

import argparse
from typing import Dict
import pandas as pd
from sqlalchemy import text

from .config import Settings, load_settings
from .db import create_pg_engine, ensure_connection
from .excel_processing import (
    choose_header_row,
    drop_empty_columns,
    infer_col,
    load_excel,
    normalize_columns,
    prepare_for_sql,
    resolve_table_name,
    pg_ident,
)
from .logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Charge un fichier Excel dans PostgreSQL")
    parser.add_argument("source", help="Chemin du fichier .xlsx/.xlsb à traiter")
    parser.add_argument(
        "--schema",
        dest="target_schema",
        default=None,
        help="Schéma cible dans PostgreSQL (défaut: public)",
    )
    parser.add_argument(
        "--if-exists",
        dest="if_exists",
        default=None,
        choices=["append", "replace", "fail"],
        help="Comportement si la table existe déjà",
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Affiche un aperçu des données normalisées",
    )
    parser.add_argument(
        "--header-scan",
        type=int,
        default=None,
        help="Nombre de lignes à scanner pour détecter l'en-tête",
    )
    return parser


def run(settings: Settings) -> int:
    logger = configure_logging()
    runtime = settings.runtime

    logger.info("Fichier source: %s", runtime.source_path)
    sheets = load_excel(runtime.source_path)
    logger.info("Feuilles détectées: %s", ", ".join(sheets.keys()))

    engine = create_pg_engine(settings.db)
    with ensure_connection(engine) as conn:
        version = conn.execute(text("select version()")).scalar()
        logger.info("Connecté à PostgreSQL")
        logger.info(version)

    total_sheets = len(sheets)
    imported = 0

    for sheet_name, raw_sheet in sheets.items():
        logger.info("\n[feuille] %s: shape initiale=%s", sheet_name, raw_sheet.shape)

        cleaned = raw_sheet.dropna(how="all", axis=0).dropna(how="all", axis=1)
        logger.info("[feuille] %s: après drop vides -> %s", sheet_name, cleaned.shape)
        if cleaned.empty:
            logger.info("[feuille] %s: vide, on passe.", sheet_name)
            continue

        header_row = choose_header_row(cleaned, scan=runtime.header_scan_rows)
        logger.info("[feuille] %s: header choisi à la ligne %s", sheet_name, header_row)

        columns = normalize_columns(cleaned.iloc[header_row].tolist())
        logger.info("[feuille] %s: colonnes normalisées -> %s", sheet_name, ", ".join(columns))

        df = cleaned.iloc[header_row + 1 :].copy()
        df.columns = columns
        df = df.dropna(how="all")
        df = drop_empty_columns(df)
        logger.info("[feuille] %s: shape après nettoyage -> %s", sheet_name, df.shape)
        if df.empty:
            logger.info("[feuille] %s: vide après normalisation, on passe.", sheet_name)
            continue

        schema_tags: Dict[str, str] = {}
        for column in df.columns:
            casted, tag = infer_col(df[column])
            df[column] = casted
            schema_tags[column] = tag

        table_name = pg_ident(resolve_table_name(runtime.source_path.stem, sheet_name, total_sheets))

        logger.info("\n--- Feuille: %s -> %s.%s ---", sheet_name, runtime.target_schema, table_name)
        logger.info("rows=%s | cols=%s", len(df), df.shape[1])
        logger.info("Colonnes: %s", ", ".join(df.columns.astype(str)))

        for column, tag in schema_tags.items():
            logger.info("  - %s: %s", column, tag)

        if runtime.show_sample:
            with pd.option_context("display.max_columns", 120, "display.width", 220):
                logger.info("\nSample (top 10):\n%s", df.head(10))

        df_sql, dtype_map = prepare_for_sql(df, schema_tags)
        try:
            df_sql.to_sql(
                name=table_name,
                con=engine,
                schema=runtime.target_schema,
                if_exists=runtime.if_exists,
                index=False,
                dtype=dtype_map,
                method="multi",
                chunksize=1000,
            )
            logger.info("[ok] Inserted into %s.%s", runtime.target_schema, table_name)
            imported += 1
        except Exception as exc:  # pragma: no cover - dépend de la base
            logger.error("[error] Échec insertion %s.%s: %s", runtime.target_schema, table_name, exc)

    logger.info("\nDone. Feuilles analysées & importées: %s", imported)
    return imported


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(
        source_path=args.source,
        target_schema=args.target_schema,
        if_exists=args.if_exists,
        show_sample=args.show_sample,
        header_scan_rows=args.header_scan,
    )
    return run(settings)


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
