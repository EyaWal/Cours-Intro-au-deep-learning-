# --- Initialisation ---
# device = ... (cuda)
# model = ImageCaptioningModel(...).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss(ignore_index=0)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# writer = SummaryWriter("runs/captioning")
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from model import ImageCaptioningModel
from dataset import train_loader, test_loader
from text_utils import (
    pretrained_embeddings,
    idx2word,
    word2idx,
    tokenize,
    untokenize
)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 15
hidden_dim = 512

# Embeddings : 300 dimensions
embedding_model = pretrained_embeddings

# Modèle
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

model = ImageCaptioningModel(
    cnn_model=resnet,
    embedding_model=embedding_model,
    lstm_hidden_dim=hidden_dim
).to(device)

# Geler le CNN pré-entraîné
for param in model.cnn.parameters():
    param.requires_grad = False

# Optimisation
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

criterion = nn.CrossEntropyLoss(
    ignore_index=word2idx["<pad>"]
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)

writer = SummaryWriter("runs/captioning")


def prepare_captions(captions):
    # Flickr30k fournit 5 légendes par image.
    # Avec le collate par défaut, captions[0]
    # contient la première légende de chaque image.
    texts = captions[0]

    sequences = [tokenize(text, word2idx) for text in texts]

    # Padding à la longueur maximale du batch
    max_len = max(len(seq) for seq in sequences)
    pad_id = word2idx["<pad>"]

    sequences = [
        seq + [pad_id] * (max_len - len(seq))
        for seq in sequences
    ]

    return torch.tensor(sequences, dtype=torch.long, device=device)
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i, (images, captions) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device)
        
        # Tokenisation manuelle du batch (simplifiée)
        caps = torch.LongTensor([tokenize(x, word2idx) for x in captions[0]]).to(device)
        
        # On fournit toutes les séquences sauf le dernier mot en entrée
        outputs = model(images, caps[:, :-1])
        
        # On s'attend à prédire la séquence décalée d'un cran (captions[:, 1:])
        # Reshape pour la Loss : (batch_size * seq_len, vocab_size)
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), caps[:, 1:].reshape(-1))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            writer.add_scalar('Loss/Train_Step', total_loss / 10, epoch * len(train_loader) + i)
            total_loss = 0

    # --- Évaluation et Génération ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, captions in test_loader:
            images = images.to(device)
            caps = torch.LongTensor([tokenize(x, word2idx) for x in captions[0]]).to(device)
            
            outputs = model(images, caps[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), caps[:, 1:].reshape(-1))
            val_loss += loss.item()

            # --- Visualisation d'une prédiction ---
            predictions = outputs[-1].argmax(dim=-1)
            pred_text = " ".join(untokenize(predictions.cpu().numpy(), idx2word))
            true_text = " ".join(untokenize(caps[-1][1:].cpu().numpy(), idx2word))
            
    print(f"Epoch {epoch + 1}")
    print(f"Prediction : {pred_text}")
    print(f"Target     : {true_text}")
    
    writer.add_scalar('Loss/Validation', val_loss / len(test_loader), epoch)
    
    scheduler.step()
    writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)