# Aerotech ETL

Ce dépôt contient un POC industrialisé pour automatiser l'extraction de fichiers Excel et le chargement dans une base PostgreSQL.

## Installation

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Configurer les variables d'environnement suivantes pour la connexion PostgreSQL :

- `PG_DB`
- `PG_USER`
- `PG_PASS`
- `PG_HOST` (optionnel, défaut : instance Render)
- `PG_PORT` (optionnel, défaut : `5432`)

## Utilisation

```bash
aerotech-extract path/to/fichier.xlsx --schema public --if-exists append --show-sample
```

Options principales :

- `--schema` : schéma PostgreSQL cible (défaut : `public`).
- `--if-exists` : comportement si la table existe (`append`, `replace`, `fail`).
- `--show-sample` : affiche un aperçu des 10 premières lignes normalisées.
- `--header-scan` : nombre de lignes à analyser pour détecter la ligne d'en-tête.

## Organisation du code

- `src/aerotech_etl/config.py` : chargement et validation de la configuration (base de données, options runtime).
- `src/aerotech_etl/db.py` : création et vérification de l'engine SQLAlchemy + mapping des types.
- `src/aerotech_etl/excel_processing.py` : fonctions de nettoyage, inférence de schéma et préparation des données.
- `src/aerotech_etl/cli.py` : CLI orchestrant la lecture Excel et l'insertion PostgreSQL.
- `src/aerotech_etl/logging_utils.py` : configuration du logger.

Les fichiers Excel de test utilisés lors du POC sont stockés dans `fic/` et peuvent servir d'exemples.
