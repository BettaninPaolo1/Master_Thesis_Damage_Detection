import os
import random
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAZIONE (Logica TRAIN_FRAME_DT)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Si assume che paolo_bridge e TRAIN_FRAME_DT siano nella stessa root
BASE_PATH = os.path.join(CURRENT_DIR, "..", "TRAIN_FRAME_DT")
ID = "2_1"

DATA_DIR_TRAIN = os.path.normpath(os.path.join(BASE_PATH, f"istantrain_{ID}"))
DATA_DIR_TEST = os.path.normpath(os.path.join(BASE_PATH, f"istantest_{ID}"))

CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEQ_LEN = 600
N_CLASS = 6
ACCELERATIONS = 0 # 0 per U_concat_, 1 per U2_concat_

# ==========================================
# 2. FUNZIONI DI UTILITÀ
# ==========================================

def RMS(signal):
    """Calcola il Root Mean Square del segnale."""
    return np.sqrt(np.mean(signal**2))

def load_and_process_data(path_data, is_training=True, added_snr=100, verbose=False):
    """
    Carica i segnali seguendo la logica di slicing del file VAE_for_MCMC.
    Aggiunto il parametro 'verbose' per stampare i controlli interni.
    """
    print(f"\n--- Caricamento Dati: {os.path.basename(path_data)} ---")
    
    # A. Caricamento Labels e Filtraggio Classe 0
    try:
        label_path_class = os.path.join(path_data, 'Damage_class.csv')
        label_path_level = os.path.join(path_data, 'Damage_level.csv')
        labels_class = np.genfromtxt(label_path_class).astype(int)
        labels_level = np.genfromtxt(label_path_level)
    except Exception as e:
        print(f"[ERRORE] Critico nel caricamento labels: {e}")
        return None, None

    # Filtra solo i casi non sani (Classe 1-6)
    valid_indices = np.where(labels_class != 0)[0]
    n_ist = len(valid_indices)
    
    if n_ist == 0:
        print("ATTENZIONE: Nessun dato trovato dopo il filtraggio Classe 0.")
        return None, None

    # B. Configurazione Percorsi e Statistiche
    prefix = "U2_concat_" if ACCELERATIONS == 1 else "U_concat_" 
    stats_dir = os.path.join(CURRENT_DIR, "models_evolution")
    if not os.path.exists(stats_dir): 
        os.makedirs(stats_dir)
    
    means_path = os.path.join(stats_dir, "signals_means.npy")
    stds_path = os.path.join(stats_dir, "signals_stds.npy")

    if is_training:
        signals_means = np.zeros(len(CHANNELS))
        signals_stds = np.zeros(len(CHANNELS))
    else:
        # Carica le medie del training se siamo in fase di test
        if os.path.exists(means_path) and os.path.exists(stds_path):
            signals_means = np.load(means_path)
            signals_stds = np.load(stds_path)
        else:
            print("[ERRORE] File statistiche mancanti per il test set!")
            return None, None

    # C. Struttura Dati
    X_noise = np.zeros((n_ist, SEQ_LEN, len(CHANNELS)))
    
    # D. Lettura e Processing Canale per Canale
    for i_ch, ch_num in enumerate(CHANNELS):
        file_path = os.path.join(path_data, f"{prefix}{ch_num}.csv")
        try:
            # Carica l'intero CSV come array 1D
            raw_data = pd.read_csv(file_path, header=None).to_numpy()[:, 0]
        except Exception as e:
            print(f"[ERRORE] Lettura file canale {ch_num}: {e}")
            return None, None
        
        for i_idx, original_idx in enumerate(valid_indices):
            # Logica di slicing precisa dal file sorgente
            start = 1 + original_idx * (1 + SEQ_LEN)
            end = (original_idx + 1) * (1 + SEQ_LEN)
            
            signal = raw_data[start:end]
            
            # --- CHECK DIAGNOSTICO: Primi valori raw ---
            if verbose and i_idx == 0 and i_ch == 0:
                print(f"   [CHECK] Primi 5 valori RAW (Canale {ch_num}, Campione Originale {original_idx}):\n   {signal[:5]}")
            
            # Corruzione con rumore Gaussiano
            rms_val = RMS(signal)
            dev_std = rms_val / np.sqrt(added_snr)
            noise = st.norm.rvs(loc=0, scale=dev_std, size=SEQ_LEN)
            X_noise[i_idx, :, i_ch] = signal + noise
            
        if is_training:
            # Calcolo statistiche per la normalizzazione
            signals_means[i_ch] = np.mean(X_noise[:, :, i_ch])
            signals_stds[i_ch] = np.std(X_noise[:, :, i_ch])

    # E. Salvataggio Statistiche e Normalizzazione Z-Score
    if is_training:
        np.save(means_path, signals_means)
        np.save(stds_path, signals_stds)

    for i_ch in range(len(CHANNELS)):
        if signals_stds[i_ch] != 0:
            X_noise[:, :, i_ch] = (X_noise[:, :, i_ch] - signals_means[i_ch]) / signals_stds[i_ch]

    # F. Creazione Vettori Label (Livello - 0.25)
    y_vector = np.zeros((n_ist, N_CLASS))
    for i_idx, original_idx in enumerate(valid_indices):
        cls_idx = labels_class[original_idx] - 1 # Mappa 1-6 su 0-5
        lvl = labels_level[original_idx]
        if 0 <= cls_idx < N_CLASS:
            y_vector[i_idx, cls_idx] = lvl - 0.25 

    print(f"   -> Caricamento completato. Shape X: {X_noise.shape}, Shape y: {y_vector.shape}")
    return X_noise, y_vector

# ==========================================
# 3. TEST VISIVO E DIAGNOSTICO
# ==========================================
if __name__ == "__main__":
    # Caricamento dati di esempio con verbose=True per vedere i dati raw
    X, y = load_and_process_data(DATA_DIR_TRAIN, is_training=True, added_snr=100, verbose=True)
    
    if X is not None and y is not None:
        print("\n" + "="*45)
        print("     DIAGNOSTICA POST-CARICAMENTO")
        print("="*45)
        
        # 1. Controllo Normalizzazione (Z-Score)
        print("\n--- 1. Verifica Normalizzazione Z-Score ---")
        for ch_idx in range(len(CHANNELS)):
            ch_data = X[:, :, ch_idx]
            mean_val = np.mean(ch_data)
            std_val = np.std(ch_data)
            print(f"Canale {CHANNELS[ch_idx]:<2}: Media = {mean_val:>8.4f} (attesa ~0.0000) | Std = {std_val:>7.4f} (attesa ~1.0000)")

        # 2. Controllo Labeling
        print("\n--- 2. Verifica Etichette (Prime 5) ---")
        for i in range(min(5, len(y))):
            pred_class = np.argmax(y[i]) + 1
            original_level = y[i, pred_class - 1] + 0.25 
            print(f"Campione {i} -> Classe {pred_class} | Livello Originale {original_level:.3f}")
            print(f"Vettore y:  {y[i]}\n")
        
        # 3. Test Visivo (Plot)
        print("--- 3. Generazione Plot ---")
        idx = random.randint(0, len(X) - 1)
        ch_to_plot = 0
        
        plt.figure(figsize=(12, 5))
        plt.plot(X[idx, :, ch_to_plot], color='#1f77b4', linewidth=1, label=f"Canale {CHANNELS[ch_to_plot]}")
        plt.title(f"Verifica Segnale - Campione {idx} (Classe Originale: {np.argmax(y[idx])+1})")
        plt.xlabel("Time Step")
        plt.ylabel("Ampiezza Normalizzata (Z-Score)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        print(f"Visualizzazione del campione {idx} completata. Controlla la finestra del grafico.")
        plt.show()
    else:
        print("\n[ERRORE] Impossibile generare la diagnostica: dati non caricati correttamente.")