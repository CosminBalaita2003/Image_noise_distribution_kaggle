import numpy as np
import pandas as pd
import os
import time
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- CONFIGURATION ---
TRAIN_CSV = 'train.csv'
VAL_CSV = 'validation.csv'
TEST_CSV = 'test.csv'
IMG_FOLDER = 'samples'
OUTPUT_FILE = 'submission_hybrid.csv'

# --- 1. FEATURE EXTRACTOR ---
def get_single_img_features(img):
    # Extracts statistical and texture features from a single image.
    pixels = img.flatten()
    feats = []
    
    # Calculate basic pixel statistics
    feats.append(np.mean(pixels))
    feats.append(np.std(pixels))
    feats.append(skew(pixels))
    feats.append(kurtosis(pixels))
    
    # Compute density histogram with 16 bins
    hist, _ = np.histogram(pixels, bins=16, range=(0, 255), density=True)
    feats.extend(hist)
    
    # Compute Fast Fourier Transform for frequency domain analysis
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-6)
    feats.append(np.mean(magnitude))
    feats.append(np.std(magnitude))
    
    # Compute Texture metrics using Gray Level Co-occurrence Matrix (GLCM)
    H, W = img.shape
    center_img = img[H//2-32:H//2+32, W//2-32:W//2+32].astype(np.uint8)
    try:
        # Distance 1 pixel
        g = graycomatrix(center_img, [1], [0, np.pi/2], levels=256, normed=True, symmetric=True)
        feats.append(graycoprops(g, 'contrast').mean())
        feats.append(graycoprops(g, 'energy').mean())
        feats.append(graycoprops(g, 'homogeneity').mean())
        feats.append(graycoprops(g, 'correlation').mean())
    except:
        feats.extend([0, 0, 0, 0])
    
    return np.array(feats)

def get_pair_features(p1, p2):
    # Combines features from two images into a single vector.
    try:
        img1 = np.load(p1).astype(np.float32)
        img2 = np.load(p2).astype(np.float32)
        
        f1 = get_single_img_features(img1)
        f2 = get_single_img_features(img2)
        
        # Calculate absolute difference between feature vectors
        diff_feats = np.abs(f1 - f2)
        
        # Compute Cosine Distance to measure distribution similarity
        try:
            cos_dist = cosine(f1, f2)
            if np.isnan(cos_dist): cos_dist = 0
        except:
            cos_dist = 0
            
        # Calculate Structural Similarity Index (SSIM)
        ssim_val = ssim(img1, img2, data_range=255.0)
        
        # Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
        mae = np.mean(np.abs(img1 - img2))
        rmse = np.sqrt(np.mean((img1 - img2)**2))
        
        final_features = np.concatenate([diff_feats, [cos_dist, ssim_val, mae, rmse]])
        return final_features

    except Exception as e:
        return np.zeros(30) 

def prepare_dataset(csv_file, img_folder, is_test=False):
    # Loads CSV and extracts features for all image pairs.
    print(f"--- Processing {csv_file} ---")
    df = pd.read_csv(csv_file)
    X = []
    y = []
    ids = []
    
    start_time = time.time()
    for i, row in df.iterrows():
        id1 = row['id_noise_1']
        id2 = row['id_noise_2']
        p1 = os.path.join(img_folder, f"{id1}.npy" if not str(id1).endswith('.npy') else id1)
        p2 = os.path.join(img_folder, f"{id2}.npy" if not str(id2).endswith('.npy') else id2)
        
        X.append(get_pair_features(p1, p2))
        
        if not is_test:
            y.append(row['label'])
        else:
            ids.append(f"({id1},{id2})")
            
        if (i+1) % 2000 == 0:
            print(f"   ... {i+1} processed")

    return np.array(X), np.array(y), ids

# --- MAIN FLOW ---
def main():
    # 1. Load and prepare training data
    X_train, y_train, _ = prepare_dataset(TRAIN_CSV, IMG_FOLDER)
    
    if os.path.exists(VAL_CSV):
        print("Merging validation data into training set...")
        X_val, y_val, _ = prepare_dataset(VAL_CSV, IMG_FOLDER)
        X_train = np.concatenate((X_train, X_val))
        y_train = np.concatenate((y_train, y_val))
    
    print(f"\nTraining Shape: {X_train.shape}")

    # 2. DEFINE HYBRID ENSEMBLE
    print("\n--- Initializing Hybrid Ensemble ---")
    
    # Initialize Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=1000,
        random_state=42,
        n_jobs=-1
    )
    
    # Initialize Histogram-based Gradient Boosting Classifier
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_depth=None,
        random_state=42
    )
    
    # Create pipeline with scaler for HGB stability
    hgb_pipe = make_pipeline(StandardScaler(), hgb)

    # Initialize Voting Classifier to combine predictions (Soft Voting)
    eclf = VotingClassifier(
        estimators=[('rf', rf), ('hgb', hgb_pipe)],
        voting='soft',
        n_jobs=-1
    )
    
    # 3. Train the ensemble model
    print("Start Training")
    eclf.fit(X_train, y_train)
    print("Hybrid Model Trained!")

    # 4. Generate predictions on test set
    print("\n--- Test Prediction ---")
    X_test, _, test_ids = prepare_dataset(TEST_CSV, IMG_FOLDER, is_test=True)
    preds = eclf.predict(X_test)
    
    # Save submission file
    df_sub = pd.DataFrame({'id_pair': test_ids, 'label': preds.astype(int)})
    df_sub.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! {OUTPUT_FILE}")

if __name__ == "__main__":
    main()