# Application Streamlit - Projet 2 GTSRB

Cette version transforme `Projet2.ipynb` en application Streamlit deployable.
Le modele sauvegarde est deja exporte dans `artifacts/traffic_sign_cnn.keras`
et l'application le charge automatiquement au demarrage.

## Ce que fait l'application

- utilise le meme pretraitement que le notebook (`32x32`, normalisation `/255`)
- reutilise la meme architecture CNN
- charge automatiquement le modele sauvegarde
- permet de chercher une image du jeu `GTSRB/Test` par nom de fichier
- affiche la prediction du modele et la compare a la classe reelle

## Fichiers importants

- `streamlit_app.py` : application Streamlit
- `traffic_sign_pipeline.py` : pipeline partage
- `train_model.py` : script pour reexporter un modele si besoin
- `prepare_deploy_bundle.py` : creation d'un bundle minimal de deploiement
- `artifacts/traffic_sign_cnn.keras` : modele charge automatiquement
- `artifacts/model_metadata.json` : metriques du modele exporte

## Installation locale

```bash
pip install -r requirements.txt
```

## Lancer l'application

Sous Windows, le plus robuste est :

```bash
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Ou apres activation de l'environnement :

```bash
python -m streamlit run streamlit_app.py
```

## Reexporter le modele

Seulement si vous voulez recalculer un nouveau modele :

```bash
python train_model.py
```

Cela mettra a jour :

- `artifacts/traffic_sign_cnn.keras`
- `artifacts/model_metadata.json`
- `artifacts/training_history.csv`

## Deploiement

Pour qu'un tiers ouvre le lien et puisse tester immediatement le modele,
le deploiement doit inclure les fichiers suivants :

- `streamlit_app.py`
- `traffic_sign_pipeline.py`
- `requirements.txt`
- le dossier `GTSRB/Test`
- le dossier `GTSRB/Meta`
- `GTSRB/Test.csv`
- `GTSRB/Meta.csv`
- le dossier `artifacts` avec `traffic_sign_cnn.keras`

Dans cette configuration, aucun entrainement n'est lance sur le serveur :
le modele est deja sauvegarde et charge automatiquement par l'application.

Si vous voulez preparer un dossier propre pour GitHub et Streamlit Cloud :

```bash
python prepare_deploy_bundle.py
```

Le dossier `deploy_bundle/` contiendra seulement les fichiers utiles a l'inference et au deploiement.
