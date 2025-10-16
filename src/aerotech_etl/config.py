"""Gestion centralisée de la configuration de l'outil d'extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

_DEFAULT_HOST = "dpg-d3jq6apr0fns738f81i0-a.frankfurt-postgres.render.com"


@dataclass(frozen=True)
class DatabaseConfig:
    """Paramètres de connexion PostgreSQL."""

    host: str
    port: int
    database: str
    user: str
    password: str
    sslmode: str = "require"

    @classmethod
    def from_env(cls, *, prefix: str = "PG", defaults: Optional[dict] = None) -> "DatabaseConfig":
        defaults = defaults or {}
        env = os.environ
        host = env.get(f"{prefix}_HOST") or defaults.get("host") or _DEFAULT_HOST
        port = int(env.get(f"{prefix}_PORT") or defaults.get("port") or 5432)
        database = env.get(f"{prefix}_DB") or defaults.get("database")
        user = env.get(f"{prefix}_USER") or defaults.get("user")
        password = env.get(f"{prefix}_PASS") or defaults.get("password")

        missing = [
            name
            for name, value in {
                "database": database,
                "user": user,
                "password": password,
            }.items()
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                "Paramètres DB manquants dans les variables d'environnement : " + joined
            )

        return cls(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            sslmode=env.get(f"{prefix}_SSLMODE", defaults.get("sslmode", "require")),
        )


@dataclass(frozen=True)
class RuntimeOptions:
    """Options contrôlant l'exécution de l'extraction."""

    source_path: Path
    target_schema: str = "public"
    if_exists: str = "append"
    show_sample: bool = False
    header_scan_rows: int = 30

    @classmethod
    def from_kwargs(
        cls,
        *,
        source_path: str | Path,
        target_schema: Optional[str] = None,
        if_exists: Optional[str] = None,
        show_sample: Optional[bool] = None,
        header_scan_rows: Optional[int] = None,
    ) -> "RuntimeOptions":
        path = Path(source_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Fichier Excel introuvable: {path}")

        if_exists_value = (if_exists or cls.if_exists).lower()
        if if_exists_value not in {"append", "replace", "fail"}:
            raise ValueError(
                "if_exists doit être l'une des valeurs suivantes: append, replace, fail"
            )

        return cls(
            source_path=path,
            target_schema=(target_schema or cls.target_schema),
            if_exists=if_exists_value,
            show_sample=bool(show_sample) if show_sample is not None else cls.show_sample,
            header_scan_rows=
            int(cls.header_scan_rows if header_scan_rows is None else header_scan_rows),
        )


@dataclass(frozen=True)
class Settings:
    """Configuration complète de l'outil."""

    db: DatabaseConfig
    runtime: RuntimeOptions
    extra: dict = field(default_factory=dict)


def load_settings(
    *,
    source_path: str | Path,
    target_schema: Optional[str] = None,
    if_exists: Optional[str] = None,
    show_sample: Optional[bool] = None,
    header_scan_rows: Optional[int] = None,
    db_defaults: Optional[dict] = None,
) -> Settings:
    """Assemble et valide les paramètres nécessaires à l'extraction."""

    runtime = RuntimeOptions.from_kwargs(
        source_path=source_path,
        target_schema=target_schema,
        if_exists=if_exists,
        show_sample=show_sample,
        header_scan_rows=header_scan_rows,
    )
    db = DatabaseConfig.from_env(defaults=db_defaults)
    return Settings(db=db, runtime=runtime)
