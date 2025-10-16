# Aerotech ETL simplifié

Ce dépôt fournit un petit script pour envoyer un classeur Excel vers une base PostgreSQL.
Il est volontairement minimaliste afin de rester facile à lire et à adapter.

## 1. Installation rapide

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Configurer la connexion PostgreSQL

Copiez l'exemple et remplissez-le avec vos identifiants :

```bash
cp .env.example .env
```

Modifiez `.env`, puis chargez les variables dans votre shell avant d'exécuter le script :

```bash
export $(grep -v '^#' .env | xargs)
```

Variables requises :

- `PG_DB`
- `PG_USER`
- `PG_PASS`

Variables optionnelles :

- `PG_HOST` (défaut : `localhost`)
- `PG_PORT` (défaut : `5432`)

## 3. Lancer une importation

```bash
aerotech-extract chemin/vers/fichier.xlsx
```

Options utiles :

- `--table` : nom de la table cible (par défaut, le nom du fichier).
- `--schema` : schéma PostgreSQL (par défaut `public`).
- `--if-exists` : `fail` (défaut), `replace` ou `append`.
- `--sheet` : nom d'une feuille précise. Sans cette option, toutes les feuilles sont envoyées.

Le script supprime simplement les lignes/colonnes entièrement vides et nettoie les noms de
colonnes pour être compatibles avec PostgreSQL.

## 4. Structure du code

Tout le comportement est concentré dans `src/aerotech_etl/cli.py`. La fonction `main()`
peut être copiée telle quelle dans un autre projet ou appelée via `python -m aerotech_etl.cli`.

## 5. Astuces pour le POC

- Testez d'abord avec l'option `--if-exists replace` sur une petite table de recette.
- Si un classeur contient plusieurs feuilles, elles sont envoyées les unes après les
  autres vers la **même** table.
- Pour adapter le nettoyage (renommer des colonnes, filtrer des lignes, etc.), modifiez
  la fonction `clean_dataframe` selon vos besoins.
