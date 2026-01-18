import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURARE ---
FILE_A = 'submission_rf3.csv'       # Pune aici submisia ta bună (benchmark)
FILE_B = 'submission_hybrid.csv'    # Pune aici submisia nouă experimentală
IMG_FOLDER = 'samples'              # Folderul cu imagini
MAX_IMAGES = 20                     # Câte perechi vrei să verifici manual

def compare_submissions():
    # 1. Încărcare CSV-uri
    if not os.path.exists(FILE_A) or not os.path.exists(FILE_B):
        print("EROARE: Nu găsesc unul dintre fișierele CSV.")
        return

    df_a = pd.read_csv(FILE_A)
    df_b = pd.read_csv(FILE_B)

    # Redenumim coloanele pentru a le putea uni
    df_a = df_a.rename(columns={'label': 'label_A'})
    df_b = df_b.rename(columns={'label': 'label_B'})

    # 2. Unim tabelele pe baza ID-ului perechii
    # Asta ne asigură că comparăm exact aceleași perechi, chiar dacă ordinea e diferită
    merged = pd.merge(df_a, df_b, on='id_pair')

    # 3. Găsim CONFLICTELE (Unde modelele nu sunt de acord)
    conflicts = merged[merged['label_A'] != merged['label_B']]
    
    total_diff = len(conflicts)
    total_rows = len(merged)
    percent = (total_diff / total_rows) * 100

    print(f"\n--- RAPORT COMPARARE ---")
    print(f"Fișier A: {FILE_A}")
    print(f"Fișier B: {FILE_B}")
    print(f"Total Perechi: {total_rows}")
    print(f"Diferențe găsite: {total_diff} ({percent:.2f}%)")
    
    if total_diff == 0:
        print("Cele două submisii sunt IDENTICE! Nu ai ce compara.")
        return

    print(f"\nSe afișează primele {MAX_IMAGES} conflicte...")
    print("Sfat: Închide fereastra graficului pentru a trece la următoarea pereche.")

    # 4. Vizualizare
    count = 0
    for idx, row in conflicts.iterrows():
        if count >= MAX_IMAGES:
            break
            
        pair_id = row['id_pair']
        pred_a = int(row['label_A'])
        pred_b = int(row['label_B'])

        # Parsăm ID-ul perechii: "(10023,10045)" -> 10023, 10045
        clean_id = pair_id.replace('(', '').replace(')', '').replace('"', '').replace("'", "")
        id1, id2 = clean_id.split(',')
        id1, id2 = id1.strip(), id2.strip()

        # Construim căile
        p1 = os.path.join(IMG_FOLDER, f"{id1}.npy" if not str(id1).endswith('.npy') else id1)
        p2 = os.path.join(IMG_FOLDER, f"{id2}.npy" if not str(id2).endswith('.npy') else id2)

        try:
            img1 = np.load(p1).astype(np.float32)
            img2 = np.load(p2).astype(np.float32)

            # Calculăm niște statistici rapide ca să te ajute să decizi
            std1, std2 = np.std(img1), np.std(img2)
            mean1, mean2 = np.mean(img1), np.mean(img2)
            
            # --- DESENARE ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Titlu General
            fig.suptitle(f"CONFLICT la {pair_id}\n"
                         f"Model A zice: {pred_a} | Model B zice: {pred_b}", 
                         fontsize=14, color='red', weight='bold')

            # Img 1
            axes[0].imshow(img1, cmap='gray')
            axes[0].set_title(f"Img 1\nStd: {std1:.2f} | Mean: {mean1:.2f}")
            axes[0].axis('off')

            # Img 2
            axes[1].imshow(img2, cmap='gray')
            axes[1].set_title(f"Img 2\nStd: {std2:.2f} | Mean: {mean2:.2f}")
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()
            
            count += 1

        except Exception as e:
            print(f"Nu am putut încărca imaginile pentru {pair_id}: {e}")

if __name__ == "__main__":
    compare_submissions()