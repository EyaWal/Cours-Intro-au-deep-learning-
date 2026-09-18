import os
import torch
import torchvision
from torchvision import transforms, datasets

# Normalisation "classique" pour CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# CHARGEMENT DES DONNÉES (à compléter)
trainset = datasets.CIFAR10(root='./data', train=_____, download=True, transform=transform)
testset  = datasets.CIFAR10(root='./data', train=_____, download=True, transform=transform)

# Ne pas monopoliser les CPUs sur Slurm
def get_num_workers(default=2, cap=4):
    try:
        n = int(os.getenv("SLURM_CPUS_PER_TASK", default))
    except Exception:
        n = default
    return max(0, min(cap, n))

num_workers = get_num_workers()

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=_____, num_workers=num_workers, pin_memory=True
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=_____, num_workers=num_workers, pin_memory=True
)
