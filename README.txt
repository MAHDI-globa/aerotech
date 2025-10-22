EXTRACTEUR – README
==================================================

0) OBJET
--------
Ce projet “extracteur” lit des fichiers sources (Excel/CSV), applique des ETL dédiés, et insère les données dans PostgreSQL, avec logs centralisés et archivage automatique (Loaded/Rejects).

1) PRÉREQUIS
------------
- Windows, macOS ou Linux
- Python 3.10+ (recommandé 3.11)
- Accès à une base PostgreSQL
- Droits d’écriture dans le dossier du projet (pour `logs/` et `Archives/`)

2) INSTALLATION RAPIDE
----------------------
### Windows PowerShell
```
cd C:\globasoft\extracteur
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "python-dotenv[cli]"
```
> Si PowerShell bloque les scripts :  
> `Set-ExecutionPolicy -Scope Process RemoteSigned`

### macOS / Linux (bash/zsh)
```
cd ~/globasoft/extracteur
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "python-dotenv[cli]"
```

3) ARBORESCENCE (SIMPLE)
------------------------
```
extracteur/
  logs/                         (créé automatiquement)
  src/
    run_jobs.py                (orchestrateur d’un run)
    watch_jobs.py              (surveille des dossiers & déclenche run_jobs)
    logging_utils.py           (logging unifié console + fichiers)
    db.py                      (connexion PostgreSQL)
    audit.py                   (ajout colonnes _loaded_at/_loaded_by)
    etl_*.py                   (fqt, fnp, spa, rpa, cdp, paae, catclients, …)
  .env.dev                     (variables BDD & app pour ENV=dev)
  .env.files.dev               (chemins des fichiers à traiter en dev)
  .env                         (facultatif: variables communes, ex: logs)
  requirements.txt
```

4) CONFIGURATION
----------------
### 4.1 Fichier `.env.<ENV>` (BDD & app)
Exemple `.env.dev` :
```
ENV=dev
PG_HOST=host.exemple
PG_PORT=5432
PG_DB=ma_base
PG_USER=mon_user
PG_PASSWORD=mon_mot_de_passe
PG_SSLMODE=require

# Logging
LOG_DIR=logs
LOG_LEVEL=INFO
LOG_ROTATE=time          # time = quotidien ; size = rotation par taille
LOG_BACKUP_COUNT=365     # conserver ~12 mois
LOG_MAX_BYTES=5242880    # utilisé si LOG_ROTATE=size
LOG_UNIFIED=1            # 1 = un seul fichier de log unifié
LOG_FILE_NAME=extracteur.log
```

### 4.2 Fichier `.env.files.<ENV>` (FICHIERS SOURCE)
Exemple `.env.files.dev` :
```
FILE_PATH_FQT=C:/globasoft/aerotech/fic/fichier qualité des trigrammes.xlsx
FILE_PATH_CDP=C:/globasoft/aerotech/fic/Conditionsdepaiement.xlsx
FILE_PATH_FNP=C:/globasoft/aerotech/fic/FNP AEC projets - source.xlsx
FILE_PATH_RPA=C:/globasoft/aerotech/fic/Relevé pointages AEB.xlsx
FILE_PATH_SPA=C:/globasoft/aerotech/fic/Suivi Projets AEC - source.xlsx
FILE_PATH_PAAE=C:/globasoft/aerotech/fic/Projets AEC AE 2025 - source.xlsx
FILE_PATH_CATCLIENTS=C:/globasoft/aerotech/fic/CatClients.xlsx
```
Règle : **seuls ces chemins EXACTS sont acceptés**. Toute autre entrée est rejetée.

5) UTILISATION (COMMANDES)
--------------------------
### 5.1 Run manuel (recommandé, charge `.env.dev`)
#### Windows
```
cd C:\globasoft\extracteur
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m dotenv -f .env.dev run -- .\.venv\Scripts\python.exe -m src.run_jobs
```

#### macOS / Linux
```
cd ~/globasoft/extracteur
source .venv/bin/activate
python -m dotenv -f .env.dev run -- python -m src.run_jobs
```

### 5.2 Watcher (surveillance continue)
#### Windows
```
cd C:\globasoft\extracteur
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m dotenv -f .env.dev run -- .\.venv\Scripts\python.exe -m src.watch_jobs
# Arrêt : Ctrl + C
```

#### macOS / Linux
```
cd ~/globasoft/extracteur
source .venv/bin/activate
python -m dotenv -f .env.dev run -- python -m src.watch_jobs
# Arrêt : Ctrl + C
```

### 5.3 Suivre les logs
#### Windows
```
Get-Content .\logs\extracteur.log -Wait
```
#### macOS / Linux
```
tail -f logs/extracteur.log
```

### 5.4 Raccourcis utiles (Windows)
Créer un `dev.ps1` à la racine :
```
# dev.ps1
. .\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m dotenv -f .env.dev run -- .\.venv\Scripts\python.exe -m src.watch_jobs
```
Puis :
```
cd C:\globasoft\extracteur
.\dev.ps1
```
Alias temporaires :
```
function runjobs   { .\.venv\Scripts\python.exe -m dotenv -f .env.dev run -- .\.venv\Scripts\python.exe -m src.run_jobs }
function watchjobs { .\.venv\Scripts\python.exe -m dotenv -f .env.dev run -- .\.venv\Scripts\python.exe -m src.watch_jobs }
```

6) RÈGLES DE TRAITEMENT
-----------------------
- Seuls les fichiers exactement listés dans `.env.files.<ENV>` sont traités.
- Extensions autorisées : `*.*` (contrôle par liste blanche de chemins).
- Un fichier doit être “stable” (taille inchangée pendant `READY_GRACE_SEC`, défaut 2s).
- **Succès** : archivage sous  
  `<dossier_source>/Archives/Loaded_YYYY-MM-DD_HHMM/<fichier>`
- **Rejets** (non attendus, extension/chemin incorrect, erreur ETL, fichiers temporaires/cachés) :  
  `<dossier_source>/Archives/Rejects_YYYY-MM-DD_HHMM/<fichier>`  
  + un fichier `<fichier>__reason.txt` expliquant la raison.
- Colonnes d’audit ajoutées à l’insert :  
  `_loaded_at` (UTC), `_loaded_by` (depuis `LOADED_BY` ou utilisateur système).

7) TABLES CIBLES (DB)
---------------------
- `IF_EXISTS_MODE` par défaut = `"replace"` (modifiable par ETL).
- Nom de table dérivé du nom de fichier/feuille (snake_case).
- Détection de types (DATE/DATETIME/INT/FLOAT/BOOL/TEXT).
- Colonnes d’audit en TEXT.

8) JOURNAUX (LOGGING)
---------------------
- Un **fichier par jour**, unifié si `LOG_UNIFIED=1` :
  - chemin : `logs/<LOG_FILE_NAME>.*` (ex: `logs/extracteur.log`)
- `LOG_ROTATE=time` → rotation quotidienne  
  `LOG_ROTATE=size` → rotation par taille (`LOG_MAX_BYTES` requis)
- Conservation : `LOG_BACKUP_COUNT` (ex: 365 jours)

9) VARIABLES UTILES
-------------------
- `ENV` → sélectionne `.env.<ENV>` et `.env.files.<ENV>`
- `LOADED_BY` → identifiant utilisateur (audit/log)
- `READY_GRACE_SEC` → délai (s) pour qu’un fichier soit “stable”
- `WATCH_POLL_INTERVAL` → intervalle de scan du watcher (s)

10) DÉPANNAGE (FAQ)
-------------------
- **“Variable d’environnement manquante …”**  
  → Ajoute la variable dans le `.env` approprié (ex: `LOG_MAX_BYTES` si `LOG_ROTATE=size`).
- **Watcher ne déclenche pas**  
  → Vérifie que les chemins `FILE_PATH_*` pointent bien vers tes dossiers d’entrée.  
  → Les fichiers temporaires (~$, .tmp, .part, .lock, …) sont ignorés.
- **Fichier rejeté**  
  → Lire `<Archives/Rejects_…>/<fichier>__reason.txt` pour comprendre.
- **Multiples fichiers de log**  
  → Mets `LOG_UNIFIED=1` et `LOG_FILE_NAME=extracteur.log`.
- **Import NumPy/Pandas cassé (Windows) :**  
  1) S’assurer qu’il **n’existe pas** de dossier/fichier nommé `numpy` ou `pandas` dans le projet (hors `.venv`).  
  2) Purger les caches :  
     `Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force`  
  3) Réinstaller propre :  
     `python -m pip install --no-cache-dir --force-reinstall numpy==2.3.4 pandas==2.3.3`  
  4) Vérifier :  
     `python -c "import sys, numpy, pandas; print(sys.executable); print(numpy.__version__, pandas.__version__)"`
- **(Dernier recours) Recréer la venv**  
  ```
  deactivate
  Remove-Item .\.venv -Recurse -Force
  python -m venv .venv
  . .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt "python-dotenv[cli]"
  ```

11) EXEMPLES D’EXÉCUTION “À MAIN NUE”
-------------------------------------
- Run manuel **sans** le CLI (si `run_jobs.py` appelle `load_dotenv(".env.dev")`) :
  ```
  # Windows
  .\.venv\Scripts\Activate.ps1
  python -m src.run_jobs

  # macOS / Linux
  source .venv/bin/activate
  python -m src.run_jobs
  ```
- Watcher idem :
  ```
  # Windows
  .\.venv\Scripts\Activate.ps1
  python -m src.watch_jobs

  # macOS / Linux
  source .venv/bin/activate
  python -m src.watch_jobs
  ```

12) REQUIREMENTS (exemple recommandé)
------------------------------------
> Pin stable ; inclut le CLI dotenv pour éviter une install séparée.
```
et_xmlfile==2.0.0
greenlet==3.2.4
numpy==2.3.4
openpyxl==3.1.5
pandas==2.3.3
psycopg2-binary==2.9.11
python-dateutil==2.9.0.post0
python-dotenv[cli]==1.1.1
pytz==2025.2
six==1.17.0
SQLAlchemy==2.0.44
typing_extensions==4.15.0
tzdata==2025.2
```

13) SÉCURITÉ / BONNES PRATIQUES
-------------------------------
- **Ne pas** commit les fichiers `.env.*` contenant des secrets.
- Utiliser un compte PostgreSQL aux **droits minimaux**.
- Sauvegarder régulièrement le schéma cible et les dossiers `Archives/`.

14) LICENCE
-----------
Projet interne / usage privé. Adapter selon vos besoins.
