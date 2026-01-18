# --- FILE: dataset.py (Varianta 256px SIMPLE) ---
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class NoisePairDataset(Dataset):
    def __init__(self, csv_file, img_folder, mode='train', target_size=256):
        self.df = pd.read_csv(csv_file)
        self.img_folder = img_folder
        # Nu mai avem nevoie de 'mode' sau 'target_size' pentru crop, 
        # dar le păstrăm ca să nu crăpe codul vechi.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id1 = row['id_noise_1']
        id2 = row['id_noise_2']
        label = row['label'] if 'label' in row else 0

        p1 = os.path.join(self.img_folder, f"{id1}.npy" if not str(id1).endswith('.npy') else id1)
        p2 = os.path.join(self.img_folder, f"{id2}.npy" if not str(id2).endswith('.npy') else id2)

        try:
            img1 = np.load(p1).astype(np.float32) / 255.0
            img2 = np.load(p2).astype(np.float32) / 255.0
        except:
            # Fallback
            img1 = np.zeros((256, 256), dtype=np.float32)
            img2 = np.zeros((256, 256), dtype=np.float32)

        # FĂRĂ CROP. Luăm imaginea întreagă (256x256).
        
        t1 = torch.from_numpy(img1).unsqueeze(0)
        t2 = torch.from_numpy(img2).unsqueeze(0)

        return t1, t2, torch.tensor(label, dtype=torch.float32)