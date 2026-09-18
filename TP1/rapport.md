# Introduction au Deep Learning
## Q1.c 
Le modèle de GPU alloué Nvidia-SMI 595.84 et le GPU name Persistence-M  
## Q1.d
```bash
scancel 1535
```
le nom du fichier logs : hello-slurm-1539.out
## Q1.g
Le ReqMem est la Ram demandé qui est 8 Go
Le MaxRSS est la valeur maximael utilisée 
## Q2.b
La version de Python utilsée est Python 3.10.21 qui a été vérifiée avec cette commande 
```bash 
python --version 
```
La commande utilisée pour avoir le chemin binaire 
```bash
which python
/mnt/hdd/homes/ewalha/miniforge3/envs/deeplearning/bin/python
```
## Q2.d
La sortie du script check_gpu.py : 
PyTorch version: 2.14.0+cu130
CUDA available: True
Device count: 1
Device 0 name: NVIDIA L4
## Q2.f 
La version de TensorBoard est  .
2.21.0 qui a été affichée avec cette commande 
```bash
tensorboard --version
```
## Exercice 3
## Q3.a
![Graphe](/Users/walhaeya/Organisation/01_École/3A/Intro Deep Learning/TP1/Q3a.jpeg)

## Q3.b
X: (N,3)
W1: (N,1)
b1: (1,t) -> diffusé en (N, t)
H:(N,t)
W2:(z,t)
b2:(1,z)-> diffusé en (N, z)
Y:(N,z)
## Q3.C
![Graphe](/Users/walhaeya/Organisation/01_École/3A/Intro Deep Learning/TP1/Q3c_3.jpeg)
## Q3.d
![Graphe](/Users/walhaeya/Organisation/01_École/3A/Intro Deep Learning/TP1/Q3d.jpeg)
## Q3.e
On utilise le règle de la chainene parce que ça permet de calculer le gradient en reliant les dérivées de chaque couche et donc propager l'erreur de la sorti jusqu'aux différentes couches 

Traiter l'ensemble total des données  est trop lourd pour la mémoire et faire  un seul exemple à la fois prend trop de temps . Le mini-batch est alors  le juste milieu : il permet de calculer plusieurs données en parallèle tout en gardant une direction d'apprentissage stable.
## Q3.f
