# Deploiement Streamlit Community Cloud

Ce projet est prepare pour un deploiement sur **Streamlit Community Cloud** avec
chargement automatique du modele sauvegarde.

## Structure minimale a deployer

```text
recherche/
├─ streamlit_app.py
├─ traffic_sign_pipeline.py
├─ requirements.txt
├─ .streamlit/
│  └─ config.toml
├─ artifacts/
│  ├─ traffic_sign_cnn.keras
│  ├─ model_metadata.json
│  └─ training_history.csv
└─ GTSRB/
   ├─ Test/
   ├─ Meta/
   ├─ Test.csv
   └─ Meta.csv
```

Le dossier `Train/` et le fichier `Train.csv` ne sont pas necessaires pour l'application deployee.

## Generer un bundle propre

Depuis la racine du projet :

```bash
python prepare_deploy_bundle.py
```

Le script cree un dossier `deploy_bundle/` avec uniquement les fichiers utiles au deploiement.

## Etapes de deploiement

### 1. Creer un depot GitHub

- cree un nouveau repository sur GitHub
- pousse le contenu du projet ou du dossier `deploy_bundle`

### 2. Verifier les fichiers presents

Ton depot doit contenir au minimum :

- `streamlit_app.py`
- `traffic_sign_pipeline.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `artifacts/traffic_sign_cnn.keras`
- `GTSRB/Test`
- `GTSRB/Meta`
- `GTSRB/Test.csv`
- `GTSRB/Meta.csv`

### 3. Connecter GitHub a Streamlit Community Cloud

- va sur `https://share.streamlit.io`
- connecte ton compte GitHub
- autorise Streamlit a acceder au repository que tu veux deployer

### 4. Creer l'application

- clique sur `Create app`
- choisis le repository GitHub
- choisis la branche a deployer
- indique `streamlit_app.py` comme fichier d'entree
- choisis si tu veux un sous-domaine personnalise

### 5. Choisir la version de Python

Dans `Advanced settings` :

- selectionne **Python 3.11**
- laisse les secrets vides si tu n'en utilises pas

### 6. Lancer le deploiement

- clique sur `Deploy`
- attends la fin de l'installation des dependances
- ouvre ensuite l'URL generee par Streamlit

## Notes pratiques

- le modele est deja sauvegarde dans `artifacts/traffic_sign_cnn.keras`
- l'application charge ce modele automatiquement au demarrage
- le dataset utile au test est d'environ 73 MB pour `GTSRB/Test` et 0.3 MB pour `GTSRB/Meta`
- le fichier modele pese environ 4.2 MB

## Sources officielles

- File structure and limitations: https://docs.streamlit.io/deploy/streamlit-community-cloud/status
- Dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- Deploy flow: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
