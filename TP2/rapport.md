# Régularisation, optimisation et métriques
## Exercice 1
## Q1.
Ceci est une mauvaise pratique parce que Le StandardScaler() calcule la moyenne et l'écart type à partir de l'ensemble du dataset donc il utilse ausssi les données de test avant l'entrainement du modèle d'où le data leakage 
## Q2.
On utilse à la place de Dataset IterableDataset qui permet de lire les données progressivement sans charger tout le datset en mémoire 
## Exercice 2
## Q1.
Avec l1_lambda=0.1 et l2_lambda=0, la loss explose au début (6.72) puis reste bloquée à 1.6340 à partir de l'epoch 2 donc le modèle n'apprend plus du tout.
## Q2. 
C'est l'argument weight_decay de optim.SGD qui permet d'appliquer cette régularisation L2 automatiquement.

## Q3.
 Le L1 pousse certains poids à devenir exactement 0, donc il fait une sorte de sélection de features . Le L2 au contraire réduit tous les poids un peu, sans jamais les annuler complètement .

 ## Exercice 3
 ## Q1.
 ![Comparaison des optimiseurs sur TensorBoard](images/tensorboard_optimizers.png)
 ## Q2.
 Adam converge le plus rapidement au début (suivi de très près par RMSprop) les deux chutent nettement plus vite que SGD et Momentum
 ## Q3. 
 Momentum converge plus vite et atteint une loss finale plus basse que SGD simple (0.56 contre 0.63) . Il accélère et stabilise la descente de gradient.

## Exercice 4
**Résultats obtenus sur le test set (modèle Adam) :**

| Métrique | Valeur |
|---|---|
| Precision | 0.7663 |
| Recall | 0.6876 |
| F1 | 0.7248 |
| AUC | 0.8029 |
## Q1.
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
## Q2.
Le Recall, parce que rater un vrai malade est plus grave en médecine que de faire un faux positif. On préfère sur-détecter plutôt que manquer un cas de maladie  réel
## Q3.
L’AUC évalue la capacité du modèle à distinguer les deux classes pour différents seuils, contrairement à la Precision et au Recall calculés ici avec un seuil fixe de 0,5. Elle donne donc une vision plus globale des performances du modèle.
