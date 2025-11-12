# Rapport d'exécution - ML App
Ce document décrit les étapes suivies pour configurer l'environnement, entraîner le modèle, et tester l'application, y compris les endpoints de l'API.
## 1. Configuration et Entraînement
Les étapes suivantes ont été réalisées pour préparer l'environnement et entraîner le modèle de classification Iris.
### a. Installation
Un environnement virtuel a été créé et les dépendances du projet ont été installées.
```bash
git clone https://github.com/AliLahbib/ML-project.git
cd ML-project
python -m venv .venv
source .venv/bin/activate  # Sur Windows utilisez `venv\Scripts\activate`
pip install -r requirements.txt
```
### b. Entraînement du modèle
Le script **train.py** a été exécuté. Ce script charge le jeu de données Iris, entraîne un modèle de régression logistique et l'évalue.
```bash
python src/train.py
```
Le modèle a atteint une précision de 0.9667 (soit 96.67%) sur l'ensemble de test et a été sauvegardé avec succès dans models/iris_classifier.pkl.

### c. Test de prédiction (local)
Le script **predict.py** a été utilisé pour charger le modèle sauvegardé et effectuer des prédictions sur des exemples de données.
```bash
python src/predict.py
```
Les prédictions se sont affichées correctement, confirmant que le modèle est fonctionnel.
```
Example 1: [5.1, 3.5, 1.4, 0.2]
Prediction: setosa
```