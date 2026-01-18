# --- FILE: train_full.py ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from dataset import NoisePairDataset
from model import SiameseNetwork
import os

# --- CONFIGURARE ---
TRAIN_CSV = 'train.csv'
VAL_CSV = 'validation.csv'
IMG_FOLDER = 'samples'
BATCH_SIZE = 16          # 256px cere memorie
LEARNING_RATE = 0.0001   # Rămânem la viteza sigură
EPOCHS = 30              # Fix 15 epoci pe tot setul

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("--- M4 GPU ACTIVAT (MPS) ---")
else:
    DEVICE = torch.device("cpu")
    print("--- CPU MODE ---")

def train():
    print("--- PREGĂTIRE FULL TRAINING (Train + Val) ---")
    
    # 1. Încărcăm ambele seturi cu modul 'train' (deci cu Random Crop activat)
    ds1 = NoisePairDataset(TRAIN_CSV, IMG_FOLDER, mode='train', target_size=256)
    
    if os.path.exists(VAL_CSV):
        ds2 = NoisePairDataset(VAL_CSV, IMG_FOLDER, mode='train', target_size=256)
        full_dataset = ConcatDataset([ds1, ds2])
        print(f" -> Date combinate: {len(full_dataset)} perechi.")
    else:
        full_dataset = ds1
        print(f" -> Doar train: {len(full_dataset)} perechi.")

    # DataLoader pe tot setul
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = SiameseNetwork().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler simplu
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"Start training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
            labels = labels.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            running_acc += (preds == labels).float().mean().item()

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / len(train_loader)
        
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{EPOCHS} [LR={lr:.6f}] | Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.2f}%")
        
        # Salvăm la fiecare epocă (safety)
        torch.save(model.state_dict(), "siamese_cnn_full2.pth")

    print("--> Antrenament complet! Model salvat ca 'siamese_cnn_full.pth'")

if __name__ == "__main__":
    train()