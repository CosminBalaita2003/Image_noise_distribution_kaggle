import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import time

# CONFIGURATION

TRAIN_CSV = 'train.csv'
VAL_CSV = 'validation.csv'
TEST_CSV = 'test.csv'
IMG_FOLDER = 'samples'
OUTPUT_FILE = 'results_with_tta.csv'
MODEL_SAVE_PATH = 'siamese_cnn_final.pth'

# Hyperparameters
BATCH_SIZE = 32          # Number of image pairs processed at once
EPOCHS = 15              # How many times to iterate through the entire dataset
LEARNING_RATE = 0.0005   # How fast the model updates its weights (lower is safer/finer)
TARGET_SIZE = 128        # The size we crop the images to (focusing on the center)

# --- Hardware Detection ---
# Logic: Prefer Apple M-series (MPS) > NVIDIA (CUDA) > CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("--- Engine: Apple M-Series GPU (MPS) ---")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("--- Engine: NVIDIA GPU (CUDA) ---")
else:
    DEVICE = torch.device("cpu")
    print("--- Engine: CPU (Slow - Not Recommended) ---")


# Dataset Class
class NoisePairDataset(Dataset):
    
    # Custom Dataset class to handle loading pairs of .npy images.
    # It handles reading the CSV, loading files, normalizing, and cropping.
    
    def __init__(self, csv_file, img_folder, mode='train'):
        self.df = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row from the dataframe
        row = self.df.iloc[idx]
        id1 = row['id_noise_1']
        id2 = row['id_noise_2']
        
        # Construct file paths (handling potential missing .npy extensions)
        p1 = os.path.join(self.img_folder, f"{id1}.npy" if not str(id1).endswith('.npy') else id1)
        p2 = os.path.join(self.img_folder, f"{id2}.npy" if not str(id2).endswith('.npy') else id2)

        try:
            # Load numpy arrays
            # divide by 255.0 to normalize pixel values to the [0, 1] range.
            img1 = np.load(p1).astype(np.float32) / 255.0
            img2 = np.load(p2).astype(np.float32) / 255.0
        except:
            img1 = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
            img2 = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)

        # --- CENTER CROP STRATEGY ---
        # the noise pattern is usually consistent across the image.
        # edges might contain artifacts or padding.
        # ensures all inputs are exacly 128x128.
        H, W = img1.shape
        if H > TARGET_SIZE and W > TARGET_SIZE:
            start_y = (H - TARGET_SIZE) // 2
            start_x = (W - TARGET_SIZE) // 2
            img1 = img1[start_y:start_y+TARGET_SIZE, start_x:start_x+TARGET_SIZE]
            img2 = img2[start_y:start_y+TARGET_SIZE, start_x:start_x+TARGET_SIZE]

        # Shape becomes: (1, 128, 128)
        t1 = torch.from_numpy(img1).unsqueeze(0)
        t2 = torch.from_numpy(img2).unsqueeze(0)

        if self.mode == 'test':
            return t1, t2
        else:
            label = row['label']
            return t1, t2, torch.tensor(label, dtype=torch.float32)

# Siamese Network Definition
# uses the same weights to process two different inputs.
# extracts a feature vector (embedding) for both images, then compares them.


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # The Convolutional Feature Extractor
        # This learns to detect edges, textures, and noise patterns.
        self.cnn = nn.Sequential(
            # Block 1: Detects simple features
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # Normalizes activation for stability
            nn.LeakyReLU(0.1),  # Activation function
            nn.MaxPool2d(2, 2), # Downsamples spatial size by half
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # Block 5: Detects complex features (Deepest layer)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
        )
        
        # Global Average Pooling: Reduces the remaining feature map to a single vector
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected layers to process the feature vector
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), # Randomly zeros neurons to prevent overfitting
            nn.Linear(256, 64)
        )
        
        # Final Classifier: Takes the difference between images and outputs a probability
        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid() # Squashes output between 0 and 1 (0 = different, 1 = same)
        )

    def forward_one(self, x):
        # Passes a single image through the CNN to get its embedding.
        x = self.cnn(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1) # Flatten to vector
        x = self.fc(x)
        return x

    def forward(self, img1, img2):

        # 1. Extract features for Image 1.
        # 2. Extract features for Image 2 (using same weights).
        # 3. Calculate absolute difference (L1 Distance).
        # 4. Predict similarity based on that difference.

        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        
        # L1 Distance (Absolute difference)
        diff = torch.abs(feat1 - feat2)
        
        output = self.classifier(diff)
        return output

# 3. TRAINING STAGE
def train_full_model():
    print("\n=== STAGE 1: TRAINING ===")
    
    # --- FULL TRAIN STRATEGY ---
    # merge validation data into training data.
    # 100% of available data for the final model.
    print("Merging train.csv and validation.csv...")
    df_train = pd.read_csv(TRAIN_CSV)
    if os.path.exists(VAL_CSV):
        df_val = pd.read_csv(VAL_CSV)
        df_full = pd.concat([df_train, df_val], ignore_index=True)
    else:
        df_full = df_train
        
    df_full.to_csv('temp_full_data.csv', index=False)
    print(f"Total training images: {len(df_full)}")

    # Initialize Dataset & Loader
    dataset = NoisePairDataset('temp_full_data.csv', IMG_FOLDER, mode='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Initialize Model, Loss, and Optimizer
    model = SiameseNetwork().to(DEVICE)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss (Standard for binary classification)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print(f"Start Training ({EPOCHS} epochs)...")
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode (enables Dropout/BatchNorm)
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for img1, img2, labels in dataloader:
            # Move data to GPU/CPU
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
            labels = labels.unsqueeze(1) # Reshape label to [Batch, 1]

            # Forward Pass
            optimizer.zero_grad()      # Reset gradients
            outputs = model(img1, img2) # Predict
            loss = criterion(outputs, labels) # Calculate error
            
            # Backward Pass
            loss.backward()            # Calculate gradients (backprop)
            optimizer.step()           # Update weights

            # Statistics
            running_loss += loss.item()
            preds = (outputs > 0.5).float() 
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(dataloader)
        acc = 100 * correct / total
        duration = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {duration:.0f}s")

    # Save the trained model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved at: {MODEL_SAVE_PATH}")
    
    # Clean up temp file
    if os.path.exists('temp_full_data.csv'):
        os.remove('temp_full_data.csv')
        
    return model

# 4. PREDICTION WITH TTA


def predict_with_tta(model):
    print("\n=== STAGE 2: PREDICTION WITH TTA ===")
    
    dataset = NoisePairDataset(TEST_CSV, IMG_FOLDER, mode='test')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model.eval() # Set model to evaluation mode (disable Dropout)
    final_preds = []
    
    print(f"Generating predictions for {len(dataset)} pairs using TTA...")
    
    with torch.no_grad(): # Disable gradient calculation for speed
        for i, (img1, img2) in enumerate(dataloader):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            
            
            # 1. Original View
            p1 = model(img1, img2)
            
            # 2. Horizontal Flip (Flip over width axis 3)
            p2 = model(torch.flip(img1, [3]), torch.flip(img2, [3]))
            
            # 3. Vertical Flip (Flip over height axis 2)
            p3 = model(torch.flip(img1, [2]), torch.flip(img2, [2]))
            
            # Average the probabilities
            avg_prob = (p1 + p2 + p3) / 3.0
            
            # Binarize: If prob > 0.5 -> Label 1, else Label 0
            batch_preds = (avg_prob > 0.5).float().cpu().numpy().flatten()
            final_preds.extend(batch_preds)
            
            if (i+1) % 20 == 0:
                print(f"   Processed batch {i+1}...")

    return np.array(final_preds).astype(int)


def save_submission(predictions):
    print("\n=== STAGE 3: SAVING CSV ===")
    
    df = pd.read_csv(TEST_CSV)
    
    # FORMATTING REQUIREMENT:

    df['id_pair'] = "(" + df['id_noise_1'].astype(str) + "," + df['id_noise_2'].astype(str) + ")"
    
    # Assign the predictions
    df['label'] = predictions
    
    # Filter only the required columns
    submission = df[['id_pair', 'label']]
    
    # Save to CSV
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS! File generated: {OUTPUT_FILE}")
    print(submission.head())

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Train the model on full data
    trained_model = train_full_model()
    
    # 2. Generate predictions using TTA
    predictions = predict_with_tta(trained_model)
    
    # 3. Save formatted submission file
    save_submission(predictions)