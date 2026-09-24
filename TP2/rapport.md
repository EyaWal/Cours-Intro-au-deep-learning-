# Régularisation, optimisation et métriques
## Exercice 1
## Q1.
Ceci est une mauvaise pratique parce que Le StandardScaler() calcule la moyenne et l'écart type à partir de l'ensemble du dataset donc il utilse ausssi les données de test avant l'entrainement du modèle d'où le data leakage 
## Q2.
On utilse à la place de Dataset IterableDataset qui permet de lire les données progressivement sans charger tout le datset en mémoire 
## Exercice 2
## Q1.