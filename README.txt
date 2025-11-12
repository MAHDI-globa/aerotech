EXTRACTEUR – README
==================================================

0) OBJET
--------
Ce projet “extracteur” lit des fichiers sources (Excel/CSV), applique des ETL dédiés,
et insère les données dans PostgreSQL, avec logs centralisés et archivage automatique
(Loaded/Rejects).

1) PRÉREQUIS
------------
- Windows, macOS ou Linux
- Python 3.11 recommandé
- Accès à une base PostgreSQL
- (Si la base n’écoute qu’en local) : accès SSH à la VM PostgreSQL pour créer un TUNNEL
- Droits d’écriture dans le dossier du projet (pour `logs/` et `Archives/`)

👉 Topologie (exemple AEC)
- VM Scripts : 10.31.202.13 (lance les ETL)
- VM Datawarehouse (PostgreSQL) : 10.31.202.12 (DB écoute sur 127.0.0.1:5432)
- Connexion DB depuis la VM Scripts via tunnel SSH (port local 15432)

2) INSTALLATION RAPIDE
----------------------
### Linux (RHEL/CentOS 9)
```bash
# Python 3.11 + venv
sudo dnf install -y python3.11 python3.11-devel python3.11-pip

# Projet + venv
cd ~/apps/aerotech
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install "python-dotenv[cli]"
```

### Windows PowerShell
```powershell
cd C:\globasoft\aerotech
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "python-dotenv[cli]"
# Si PowerShell bloque : Set-ExecutionPolicy -Scope Process RemoteSigned
```

3) ARBORESCENCE (SIMPLE)
------------------------
```
aerotech/
  logs/                         (créé automatiquement)
  src/
    run_jobs.py                (orchestrateur d’un run)
    watch_jobs.py              (surveille un dossier & déclenche run_jobs)
    logging_utils.py           (logging unifié)
    db.py                      (connexion PostgreSQL)
    audit.py                   (_loaded_at/_loaded_by)
    etl_*.py                   (fqt, fnp, spa, rpa, cdp, paae, catclients, …)
  .env.<ENV>                   (BDD & app)
  .env.files.<ENV>             (chemins des fichiers source)
  requirements.txt
```

4) CONFIGURATION
----------------
### 4.1 `.env.<ENV>` (BDD & app)
Exemple `.env.rec` (avec tunnel SSH activé) :
```
ENV=rec
PG_HOST=127.0.0.1
PG_PORT=15432
PG_DB=aerotec_datawarehouse
PG_USER=adminglobasoft
PG_PASSWORD=********
PG_SSLMODE=disable

# Logging
LOG_DIR=logs
LOG_LEVEL=INFO
LOG_ROTATE=time
LOG_BACKUP_COUNT=365
LOG_MAX_BYTES=5242880
LOG_UNIFIED=1
LOG_FILE_NAME=extracteur.log
```

### 4.2 `.env.files.<ENV>` (FICHIERS SOURCE)
```
FILE_PATH_FQT=/chemin/fic/fichier qualité des trigrammes.xlsx
FILE_PATH_CDP=/chemin/fic/Conditionsdepaiement.xlsx
FILE_PATH_FNP=/chemin/fic/FNP AEC projets - source.xlsx
FILE_PATH_RPA=/chemin/fic/Relevé pointages AEB.xlsx
FILE_PATH_SPA=/chemin/fic/Suivi Projets AEC - source.xlsx
FILE_PATH_PAAE=/chemin/fic/Projets AEC AE 2025 - source.xlsx
FILE_PATH_CATCLIENTS=/chemin/fic/CatClients.xlsx
```
Règle : **seuls ces chemins EXACTS sont acceptés**. Toute autre entrée est rejetée.

5) TUNNEL SSH VERS POSTGRESQL (si DB locale à la VM)
-----------------------------------------------------
Objectif : exposer `127.0.0.1:5432` de la VM DB (`10.31.202.12`) sur `127.0.0.1:15432` de la VM Scripts.

### 5.1 Ouvrir le tunnel (VM Scripts)
```bash
# tuer un éventuel tunnel existant
for p in $(ss -ltnp | awk '/:15432/ {match($7,/pid=([0-9]+)/,m); if (m[1]) print m[1]}'); do kill -9 "$p" 2>/dev/null || true; done

# ouvrir le tunnel
ssh -fN -o ExitOnForwardFailure=yes   -o ServerAliveInterval=30 -o ServerAliveCountMax=3   -L 15432:127.0.0.1:5432   mahdi.zouaoui@10.31.202.12

# vérifier
ss -ltnp | grep 15432 || echo "Tunnel KO"
```

### 5.2 Health-check du tunnel
```bash
# variables chargées ?
env | grep -E '^PG_(HOST|PORT|DB|USER|PASSWORD|SSLMODE)='

# test Python
python - <<'PY'
import os, psycopg2
conn = psycopg2.connect(
  host=os.getenv("PG_HOST","127.0.0.1"),
  port=int(os.getenv("PG_PORT","15432")),
  dbname=os.getenv("PG_DB","aerotec_datawarehouse"),
  user=os.getenv("PG_USER","adminglobasoft"),
  password=os.getenv("PG_PASSWORD",""),
  connect_timeout=5
)
cur = conn.cursor()
cur.execute("select now(), current_user")
print("DB OK ->", cur.fetchone())
conn.close()
PY
```
Si ça répond `DB OK -> (...)`, la connectivité est bonne et les scripts peuvent tourner.

### 5.3 Dépannage tunnel
- **Address already in use (15432)** : un ancien tunnel écoute. Tuer le PID (cf. commande de kill ci-dessus).
- **CLOSE-WAIT persistants** : ils disparaissent après kill du PID SSH et une minute d’attente.
- **Mot de passe demandé trop souvent** : configure une clé SSH avec `ssh-copy-id` ou un `~/.ssh/config`.

(Optionnel) service systemd pour démarrer le tunnel au boot :
```
# /etc/systemd/system/pg-tunnel.service
[Unit]
Description=SSH tunnel to PostgreSQL
After=network-online.target
Wants=network-online.target

[Service]
User=mahdi.zouaoui
ExecStart=/usr/bin/ssh -N -L 15432:127.0.0.1:5432 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 mahdi.zouaoui@10.31.202.12
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```
sudo systemctl daemon-reload
sudo systemctl enable --now pg-tunnel
systemctl status pg-tunnel
```

6) UTILISATION (COMMANDES)
--------------------------
### 6.1 Run manuel
```bash
cd ~/apps/aerotech
source .venv311/bin/activate

# charger les .env
set -a
. ./.env.rec
. ./.env.files.rec
set +a

# s'assurer que le tunnel tourne (voir §5)
# puis lancer
ENV=rec python -m src.run_jobs
```

### 6.2 Watcher (surveillance continue)
```bash
cd ~/apps/aerotech
source .venv311/bin/activate

set -a
. ./.env.rec
. ./.env.files.rec
set +a

ENV=rec python -m src.watch_jobs --env rec
# Arrêt : Ctrl + C
```

### 6.3 Suivre les logs
```bash
tail -f logs/extracteur.log
```

7) RÈGLES DE TRAITEMENT
-----------------------
- Seuls les fichiers exactement listés dans `.env.files.<ENV>` sont traités.
- Un fichier doit être “stable” (taille inchangée pendant `READY_GRACE_SEC`, défaut 2s).
- Succès : `<source>/Archives/Loaded_YYYY-MM-DD_HHMM/<fichier>`
- Rejets : `<source>/Archives/Rejects_YYYY-MM-DD_HHMM/<fichier>` + `<fichier>__reason.txt`
- Colonnes d’audit : `_loaded_at` (UTC), `_loaded_by`.

8) TABLES CIBLES (DB)
---------------------
- `IF_EXISTS_MODE` par défaut = "replace" (modifiable par ETL).
- Nom de table dérivé du nom de fichier/feuille (snake_case).
- Détection de types (DATE/DATETIME/INT/FLOAT/BOOL/TEXT).
- Colonnes d’audit en TEXT.

9) JOURNAUX (LOGGING)
---------------------
- Un fichier par jour, unifié si `LOG_UNIFIED=1`.
- Rotation : `LOG_ROTATE=time` (quotidienne) ou `size` (taille `LOG_MAX_BYTES`).
- Rétention : `LOG_BACKUP_COUNT` (ex: 365).

10) DÉPANNAGE (FAQ)
-------------------
- Tunnel OK mais “auth ident” côté DB  
  → On passe par TCP avec mot de passe. Laisse `PG_SSLMODE=disable` si réseau interne,
    sinon active `require` et mets un certificat si dispo.
- "No module named pandas/numpy"  
  → `python -m pip install --no-cache-dir --force-reinstall pandas numpy`
- `requirements.txt` incompatible avec ta version Python  
  → Utilise Python 3.11 (recommandé). En 3.9, épingle `numpy==2.0.2` et `pandas==2.2.2`.
- `Address already in use` sur 15432  
  → Tuer l’ancien tunnel (cf. §5.3).
- `psql` introuvable (optionnel)  
  → `sudo dnf install -y postgresql` puis tester :  
    `PGPASSWORD=*** psql -h 127.0.0.1 -p 15432 -U adminglobasoft -d aerotec_datawarehouse -c "select 1;"`

11) REQUIREMENTS (référence Python 3.11)
----------------------------------------
Pins stables compatibles Python 3.11 :
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

12) SÉCURITÉ / BONNES PRATIQUES
-------------------------------
- Ne pas commit les fichiers `.env.*` (secrets).
- Utiliser un compte PostgreSQL aux droits minimaux.
- Sauvegarder régulièrement le schéma cible et `Archives/`.
- Option : mettre le venv hors du repo (`~/.venvs/aerotech311`) pour éviter qu’un `git clean -fdx` le supprime.

13) LICENCE
-----------
Projet interne / usage privé. Adapter selon vos besoins.
