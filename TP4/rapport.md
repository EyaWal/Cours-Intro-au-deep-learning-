# TP4 : Réseaux récurrents et génération de séquences
## Exercice 1
### Q1. Chargement et découverte du jeu de données

Après exécution de `train_music.py`, on obtient :
- **Train :** 214 122 partitions
- **Validation :** 2 162 partitions

Voici la première partition du jeu d'entraînement :

```text
X:1
L:1/8
M:4/4
K:Emin
|: E2 EF E2 EF | DEFG AFDF | E2 EF E2 B2 |1 efe^d e2 e2 :|2 efe^d e3 B |: e2 ef g2 fe |
defg afdf |1 e2 ef g2 fe | efe^d e3 B :|2 g2 bg f2 af | efe^d e2 e2 ||
```

En regardant les premières lignes, on retrouve les informations suivantes :

- `X:1` : numéro de la partition.
- `L:1/8` : durée par défaut des notes 
- `M:4/4` : mesure à 4 temps.
- `K:Emin` : tonalité de mi mineur.

Ensuite, les lettres représentent les notes, les chiffres indiquent leur durée et les symboles `|` séparent les mesures. Les symboles `|:` et `:|` indiquent les reprises.
### Q2. Construction du vocabulaire et Padding

**1. Statistiques**

Après exécution du code, on obtient :
- Taille du vocabulaire : 95 caractères uniques
- Longueur moyenne : 290,1 caractères
- Longueur maximale : 2968 caractères

**2. Pourquoi utiliser `sorted()` ?**

On utilise `sorted()` pour avoir toujours le même ordre des caractères et donc les mêmes indices. Sans ça, les indices pourraient changer entre deux exécutions. Si on recharge un modèle déjà entraîné, il risque de ne plus associer les bons caractères aux bons indices, ce qui fausserait la génération.

**3. Pourquoi choisir `SEQ_LENGTH = 500` ?**

On choisit 500 plutôt que 2968 pour éviter d'ajouter trop de padding, sachant que la longueur moyenne est de 290 caractères. Avec 2968, on aurait beaucoup d'espaces inutiles, ce qui augmenterait le temps d'entraînement et l'utilisation de la mémoire GPU.
### Exercice 2 — Dataset PyTorch et modélisation

#### Q1. MusicDataset

Avec `SEQ_LENGTH = 500` et `BATCH_SIZE = 128`, les tenseurs `x` et `y` ont tous les deux une dimension de **(128, 499)**.

On utilise `torch.long` car les caractères sont représentés par des indices entiers qui seront utilisés par la couche `nn.Embedding`. Le type `float32` ne convient pas pour indexer cette couche.
#### Q2. Architecture RNN et LSTM

**1. Calcul du nombre de paramètres**

Pour `vocab_size = 95`, `embedding_dim = 256` et `hidden_size = 512` :

**Embedding :**

\[
95 \times 256 = 24\,320
\]

**RNN :**

\[
512 \times 256 + 512 \times 512 + 2 \times 512
= 394\,240
\]

**LSTM :**

\[
4 \times (512 \times 256 + 512 \times 512 + 2 \times 512)
= 1\,576\,960
\]

**Couche linéaire :**

\[
512 \times 95 + 95 = 48\,735
\]

On obtient donc au total :
- RNN : **467 295 paramètres**
- LSTM : **1 650 015 paramètres**

Le LSTM possède 4 fois plus de paramètres dans sa couche récurrente car il utilise 4 portes (oubli, entrée, sortie et candidate), alors que le RNN classique n'effectue qu'un seul calcul d'état caché.

**2. Pourquoi utiliser `nn.Embedding` ?**

On utilise `nn.Embedding` pour représenter chaque caractère par un vecteur de valeurs apprises. Cela permet au modèle d'apprendre des représentations plus adaptées que de simples indices numériques.

**3. Différence entre les états cachés du RNN et du LSTM**

Le RNN retourne un seul état caché `h_n`, qui contient les informations retenues après le traitement de la séquence.

Le LSTM retourne un tuple `(h_n, c_n)` :
- `h_n` : état caché, qui représente les informations utilisées en sortie.
- `c_n` : état de cellule, qui permet de conserver des informations sur une plus longue durée.

Cela permet au LSTM de mieux gérer les dépendances à long terme 