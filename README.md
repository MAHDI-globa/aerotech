# Mini import Excel → PostgreSQL

Projet réduit au strict nécessaire pour charger un classeur Excel dans PostgreSQL.

## 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Variables PostgreSQL

```bash
cp .env.example .env
```

Renseignez vos accès dans `.env`, puis exportez-les :

```bash
export $(grep -v '^#' .env | xargs)
```

Variables obligatoires : `PG_DB`, `PG_USER`, `PG_PASS`.
Variables optionnelles : `PG_HOST` (= `localhost`), `PG_PORT` (= `5432`).

## 3. Utilisation

```bash
aerotech-extract chemin/vers/fichier.xlsx
```

Options (toutes facultatives) :

* `--table` – nom de la table cible.
* `--schema` – schéma PostgreSQL (`public` par défaut).
* `--if-exists` – `fail`, `replace` ou `append`.
* `--sheet` – nom d'une feuille précise à importer.

Fonctionnement :

1. Lecture d'une ou toutes les feuilles.
2. Suppression des lignes et colonnes entièrement vides.
3. Passage des noms de colonnes en minuscules avec `_` pour les espaces.
4. Insertion dans PostgreSQL via `pandas.DataFrame.to_sql`.

Tout le code se trouve dans `src/aerotech_etl/cli.py`.
