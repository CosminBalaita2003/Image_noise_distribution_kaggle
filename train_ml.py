# --- FILE: make_submission_ml.py ---
import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURARE ---
TRAIN_CSV = 'train.csv'
VAL_CSV = 'validation.csv'
TEST_CSV = 'test.csv'         # Fișierul de concurs
IMG_FOLDER = 'samples'        # Folderul cu .npy
OUTPUT_FILE = 'submission_ml.csv'

def extract_features(img_path):
    """ Aceeași funcție ca la antrenare (Stats + Histograme) """
    try:
        img = np.load(img_path).astype(np.float32)
        pixels = img.flatten()
        
        # Statistici
        stats = [np.mean(pixels), np.std(pixels), skew(pixels), kurtosis(pixels)]
        
        # Histograma (10 bins)
        hist, _ = np.histogram(pixels, bins=10, range=(0, 255), density=True)
        
        return stats + list(hist)
    except Exception:
        return [0.0] * 14

def prepare_data(df, img_folder, has_labels=True):
    X = []
    y = []
    ids = [] # Păstrăm ID-urile pentru verificare
    
    print(f" -> Procesăm {len(df)} imagini...")
    
    count = 0
    for _, row in df.iterrows():
        id1, id2 = row['id_noise_1'], row['id_noise_2']
        
        path1 = os.path.join(img_folder, f"{id1}.npy" if not str(id1).endswith('.npy') else id1)
        path2 = os.path.join(img_folder, f"{id2}.npy" if not str(id2).endswith('.npy') else id2)
        
        f1 = extract_features(path1)
        f2 = extract_features(path2)
        
        # Diferența absolută
        diff = np.abs(np.array(f1) - np.array(f2))
        X.append(diff)
        
        if has_labels:
            label = row['label'] if 'label' in row else 0
            y.append(label)
            
        count += 1
        if count % 2000 == 0:
            print(f"    ... {count} gata")
            
    if has_labels:
        return np.array(X), np.array(y)
    else:
        return np.array(X) # Pentru test returnăm doar X

def main():
    # 1. Încărcăm Train și Validation
    print("1. Încărcăm datele de antrenare (Train + Val)...")
    df_train = pd.read_csv(TRAIN_CSV)
    
    if os.path.exists(VAL_CSV):
        df_val = pd.read_csv(VAL_CSV)
        # Le unim pentru a avea mai multe date
        df_full_train = pd.concat([df_train, df_val], ignore_index=True)
        print(f"   Total date antrenare: {len(df_full_train)}")
    else:
        df_full_train = df_train

    # 2. Extragem trăsăturile
    X_train, y_train = prepare_data(df_full_train, IMG_FOLDER, has_labels=True)

    # 3. Antrenăm Modelul
    print("\n2. Antrenăm Random Forest pe TOATE datele...")
    clf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_leaf=5, 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("   Model antrenat cu succes!")

    # 4. Procesăm Testul
    print("\n3. Procesăm fișierul TEST...")
    if not os.path.exists(TEST_CSV):
        print("EROARE: Nu găsesc test.csv!")
        return

    df_test = pd.read_csv(TEST_CSV)
    X_test = prepare_data(df_test, IMG_FOLDER, has_labels=False)

    # 5. Facem Predicții
    print("\n4. Generăm predicțiile...")
    predictions = clf.predict(X_test) # Returnează 0 sau 1

    # 6. Salvăm fișierul CSV
    # Structura trebuie să fie identică cu sample_submission.csv
    # De obicei: id_noise_1, id_noise_2, label
    
    submission = df_test.copy()
    submission['label'] = predictions
    
    # Salvăm
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS! Fișier generat: {OUTPUT_FILE}")
    print("Încarcă acest fișier pe Kaggle/Platformă.")

if __name__ == "__main__":
    main()