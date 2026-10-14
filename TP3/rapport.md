# Réseaux convolutifs
## Exercice 1
### Q1. 
Les statistiques doivent être produits uniquement sur sur le jeu d'entrainement pour ne pas avoir du data leakage à partuir des données de valid
### Q2.
On utilise RandomResizedCrop et RandomHorizontalFlip sur le jeu d'entraînement  parce que l'augmentattion est aléatoire . En effet, 
on  créer des points d’entraînement supplémentaires “gratuitement” en transformant les données
déjà existantes.

Pendant la validation, Resize et CenterCrop sont utilisés pour obtenir des images de taille uniforme avec un prétraitement déterministe . On n'applique pas de pré-processing aléatoire afin de garantir une évaluation cohérente et reproductible des performances du modèle.
### Q3.
Cette transformation peut ne pas être valide  pour du texte, des panneaux ou même des gestes sportifs asymétriques.
## Exercice 2 
### Q1.


Part couche 1 = 39 574 528/39 897 320 = 0.99
Donc elle concentre plus de 99% des paramètres 
### Q2.
Le flatten écrase la structure spatiale de l’entrée.
Dans une image, l’aspect spatial est très important, on ne veut pas avoir à le réapprendre.
### Etape 2 : AlexNet
### Q2.b
#### Q2.1
 

 
| Couche | Entrée (C, H×W) | Hyperparamètres | Sortie (C, H×W) | # Paramètres (avec biais) | Mémoire sortie (KB) |
| --- | --- | --- | --- | --- | --- |
| **Conv1** + ReLU | 3, 227×227 | 64 filtres, K=11×11, S=4, P=2 | 64, 56×56 | 23 296 | 784 |
| **MaxPool1** | 64, 56×56 | K=3×3, S=2, P=0 | 64, 27×27 | 0 | 182,25 |
| **Conv2** + ReLU | 64, 27×27 | 192 filtres, K=5×5, S=1, P=2 | 192, 27×27 | 307 392 | 546,75 |
| **MaxPool2** | 192, 27×27 | K=3×3, S=2, P=0 | 192, 13×13 | 0 | 126,75 |
| **Conv3** + ReLU | 192, 13×13 | 384 filtres, K=3×3, S=1, P=1 | 384, 13×13 | 663 936 | 253,5 |
| **Conv4** + ReLU | 384, 13×13 | 256 filtres, K=3×3, S=1, P=1 | 256, 13×13 | 884 992 | 169 |
| **Conv5** + ReLU | 256, 13×13 | 256 filtres, K=3×3, S=1, P=1 | 256, 13×13 | 590 080 | 169 |
| **MaxPool3** | 256, 13×13 | K=3×3, S=2, P=0 | 256, 6×6 | 0 | 36 |
| **Flatten** + Dropout | 256, 6×6 | p = 0,5 | 9216 (1D) | 0 | 36 |
| **Linear1** + ReLU + Dropout | 9216 | 4096 neurones (p = 0,5) | 4096 | 37 752 832 | 16 |
| **Linear2** + ReLU | 4096 | 4096 neurones | 4096 | 16 781 312 | 16 |
| **Linear3** (Sortie) | 4096 | 1000 classes | 1000 | 4 097 000 | 3,91 |
| **Total** | | | | **61 100 840** | **≈ 2 339** |


#### Q2.2

Les paramètres sont principalement concentrés dans les couches denses, surtout Linear1 et Linear2.
La mémoire des activations est principalement consommée par les premières couches convolutives Conv1 et Conv2 
#### Q2.3
AlexNet nécessite environ 2 339 KB d'activations de sortie par image
Activations du MLP  : 256 + 256 + 1000 = 1512 valeurs × 4 octets ≈ 5,9 KB par image.
Le MLP stocke moins d'activations après son flatten  et ses couches cachées, mais perd la structure spatiale de l'image. Ce paradoxe s'explique par la différence entre le nombre de paramètres entraînables et le nombre de valeurs intermédiaires produites par les couches.

Valeurs trouvées au lancement du train_imagenet.py d'après :
 
```
MLP     : 39,897,320 paramètres
AlexNet : 61,100,840 paramètres
```