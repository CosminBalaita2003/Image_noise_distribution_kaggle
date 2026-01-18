# --- FILE: make_submission_tta_final.py ---
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURARE ---
TEST_CSV = 'test.csv'
IMG_FOLDER = 'samples'
# <--- NUMELE MODELULUI TĂU CÂȘTIGĂTOR (Verifică să fie cel de 71%)
MODEL_PATH = 'siamese_cnn_v2.pth' 
OUTPUT_FILE = 'submission_TTA_formatted.csv'
BATCH_SIZE = 32

# Detectare M4 / CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("--- Engine: M4 GPU (MPS) ---")
else:
    DEVICE = torch.device("cpu")
    print("--- Engine: CPU ---")

# --- 1. ARHITECTURA (Ca să fim siguri că e cea corectă - V2) ---
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        x = self.cnn(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        diff = torch.abs(feat1 - feat2)
        output = self.classifier(diff)
        return output

# --- 2. DATASET ---
class TestDataset(Dataset):
    def __init__(self, csv_file, img_folder):
        self.df = pd.read_csv(csv_file)
        self.img_folder = img_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id1 = row['id_noise_1']
        id2 = row['id_noise_2']
        
        p1 = os.path.join(self.img_folder, f"{id1}.npy" if not str(id1).endswith('.npy') else id1)
        p2 = os.path.join(self.img_folder, f"{id2}.npy" if not str(id2).endswith('.npy') else id2)

        try:
            img1 = np.load(p1).astype(np.float32) / 255.0
            img2 = np.load(p2).astype(np.float32) / 255.0
        except:
            img1 = np.zeros((128, 128), dtype=np.float32)
            img2 = np.zeros((128, 128), dtype=np.float32)

        # Center Crop 128x128
        H, W = img1.shape
        target = 128
        if H > target and W > target:
            top = (H - target) // 2
            left = (W - target) // 2
            img1 = img1[top:top+target, left:left+target]
            img2 = img2[top:top+target, left:left+target]
        
        t1 = torch.from_numpy(img1).unsqueeze(0)
        t2 = torch.from_numpy(img2).unsqueeze(0)
        return t1, t2

def make_submission():
    # 1. Încărcare Model
    print(f"1. Încărcăm modelul: {MODEL_PATH}")
    model = SiameseNetwork().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("   Model încărcat cu succes!")
    except FileNotFoundError:
        print(f"EROARE: Nu găsesc '{MODEL_PATH}'! Verifică numele.")
        return
    
    # IMPORTANT: Evaluare mode (fără dropout)
    model.eval()

    # 2. Procesare cu TTA
    test_ds = TestDataset(TEST_CSV, IMG_FOLDER)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    final_predictions = []
    print(f"2. Generăm predicții TTA (Original + HFlip + VFlip) pentru {len(test_ds)} perechi...")

    with torch.no_grad():
        for i, (img1, img2) in enumerate(test_loader):
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            
            # --- TTA LOGIC ---
            # 1. Original
            p1 = model(img1, img2)
            
            # 2. Flip Orizontal (Axa 3 este lățimea)
            p2 = model(torch.flip(img1, [3]), torch.flip(img2, [3]))
            
            # 3. Flip Vertical (Axa 2 este înălțimea)
            p3 = model(torch.flip(img1, [2]), torch.flip(img2, [2]))
            
            # Medie aritmetică a probabilităților
            avg_prob = (p1 + p2 + p3) / 3.0
            
            # Decizia Finală
            preds = (avg_prob > 0.5).float().cpu().numpy().flatten()
            final_predictions.extend(preds)
            
            if (i+1) % 20 == 0:
                print(f"   Batch {i+1} complet...")

    # --- 3. FORMATOREA (uuid,uuid) ---
    print("\n3. Formatăm CSV-ul cu paranteze (id1,id2)...")
    df = pd.read_csv(TEST_CSV)
    
    # Construim stringul corect
    df['id_pair'] = "(" + df['id_noise_1'].astype(str) + "," + df['id_noise_2'].astype(str) + ")"
    
    # Adăugăm predicțiile
    df['label'] = np.array(final_predictions).astype(int)
    
    # Selectăm și Salvăm
    submission_df = df[['id_pair', 'label']]
    submission_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"SUCCESS! Fișier generat: {OUTPUT_FILE}")
    print("\n--- Verificare Primele 3 rânduri ---")
    print(submission_df.head(3))

if __name__ == "__main__":
    make_submission()