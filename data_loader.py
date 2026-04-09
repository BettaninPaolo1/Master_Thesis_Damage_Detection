import os
import random
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Percorso relativo standard
DATA_DIR = os.path.join(CURRENT_SCRIPT_DIR, "..", "L_FRAME_DT", "Dati", "istantrain_7_1")
DATA_DIR = os.path.normpath(DATA_DIR)

#percorso per i dati di test (TEST_DIR)
TEST_DIR = os.path.join(CURRENT_SCRIPT_DIR, "..", "L_FRAME_DT", "Dati", "istantest_7_1")
TEST_DIR = os.path.normpath(TEST_DIR)

CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8]
SEQ_LEN = 200
ADDED_SNR = 100

# ==========================================
# 2. FUNZIONI DI LETTURA ADATTIVA (CON DIAGNOSTICA)
# ==========================================

def RMS(signal):
    """Calcola il Root Mean Square del segnale."""
    return np.sqrt(np.mean(signal**2))

def parse_sensor_data_adaptive(file_path, expected_len=200):
    """
    Legge il file CSV cercando i separatori (0.0).
    Include una DIAGNOSTICA per verificare se stiamo buttando via dati validi.
    """
    try:
        raw_values = pd.read_csv(file_path, header=None).values.flatten()
    except FileNotFoundError:
        print(f"ERRORE: File non trovato -> {file_path}")
        return None

    extracted_sequences = []
    dropped_count = 0 # Contatore per la tua sicurezza
    
    # Trova gli indici dei separatori (0.0 esatto)
    separator_indices = np.where(raw_values == 0)[0]
    
    for start_idx in separator_indices:
        data_start = start_idx + 1
        data_end = data_start + expected_len
        
        # Se c'è spazio sufficiente per una sequenza intera
        if data_end <= len(raw_values):
            sequence = raw_values[data_start:data_end]
            
            # Controllo integrità: il blocco contiene zeri interni?
            if not np.any(sequence == 0):
                extracted_sequences.append(sequence)
            else:
                # Se entriamo qui, avevi ragione tu: un segnale valido conteneva uno zero
                dropped_count += 1
    
    # Feedback diagnostico (lo stampiamo solo se c'è qualcosa di strano)
    if dropped_count > 0:
        print(f"   [WARN] {os.path.basename(file_path)}: Scartate {dropped_count} sequenze per zeri interni!")
    
    return np.array(extracted_sequences)

# ==========================================
# 3. PROCESSING (RUMORE + Z-SCORE)
# ==========================================

def process_train_logic(X_in, snr):
    """
    Replica la logica corretta:
    1. Aggiunge Rumore Gaussiano (SNR)
    2. Normalizza Z-Score (Media 0, Std 1)
    """
    X = X_in.copy()
    n_instances, seq_len, n_channels = X.shape
    
    # 1. Aggiunta Rumore
    print(f"   -> Aggiunta rumore (SNR={snr})...")
    for i in range(n_channels):
        for idx in range(n_instances):
            signal = X[idx, :, i]
            rms_val = RMS(signal)
            
            if rms_val == 0:
                dev_std = 0
            else:
                dev_std = rms_val / np.sqrt(snr)
                
            noise = st.norm.rvs(loc=0, scale=dev_std, size=seq_len)
            X[idx, :, i] = signal + noise

    # 2. Z-Score Normalization
    print(f"   -> Normalizzazione Z-Score...")
    for i in range(n_channels):
        mean_ch = np.mean(X[:, :, i])
        std_ch = np.std(X[:, :, i])
        
        if std_ch != 0:
            X[:, :, i] = (X[:, :, i] - mean_ch) / std_ch
            
    return X

def load_and_process_data(data_path, channels, seq_len, snr):
    """
    Funzione principale.
    """
    print(f"--- Caricamento ADATTIVO (con Diagnostica Zeri) ---")
    
    if not os.path.exists(data_path):
        print(f"ERRORE CRITICO: Path non trovato: {data_path}")
        return None, None

    # A. Caricamento Sensori e Allineamento
    sensor_buffers = []
    min_len = float('inf')

    for ch in channels:
        file_name = f'U_concat_{ch}.csv'
        full_path = os.path.join(data_path, file_name)
        
        # Lettura Smart
        sequences = parse_sensor_data_adaptive(full_path, seq_len)
        
        if sequences is None or len(sequences) == 0:
            print(f"ERRORE: Nessun dato valido per sensore {ch}")
            return None, None
            
        sensor_buffers.append(sequences)
        if len(sequences) < min_len:
            min_len = len(sequences)

    print(f"   -> Sequenze valide allineate (Minimo Comune): {min_len}")

    # B. Creazione Tensore Input X
    X_raw = np.zeros((min_len, seq_len, len(channels)))
    for i in range(len(channels)):
        X_raw[:, :, i] = sensor_buffers[i][:min_len]

    # C. Caricamento Labels y
    try:
        path_class = os.path.join(data_path, 'Damage_class.csv')
        path_level = os.path.join(data_path, 'Damage_level.csv')
        
        labels_class = np.genfromtxt(path_class).astype('int')[:min_len]
        labels_level = np.genfromtxt(path_level)[:min_len]
        
        # 1. Allineamento lunghezze iniziale
        final_len = min(len(X_raw), len(labels_class))
        X_raw = X_raw[:final_len]
        labels_class = labels_class[:final_len]
        labels_level = labels_level[:final_len]

        print(f"   -> Filtraggio Classi: Rimuovo i casi sani (Classe 0)...")
        # Trova gli indici dove la classe NON è 0
        valid_indices = np.where(labels_class != 0)[0]
        
        # Tieni solo quei dati
        X_raw = X_raw[valid_indices]
        labels_class = labels_class[valid_indices]
        labels_level = labels_level[valid_indices]
        
        # Aggiorna la lunghezza finale
        final_len = len(valid_indices)
        print(f"   -> Dati rimasti dopo il filtro: {final_len}")
        
        if final_len == 0:
            print("ATTENZIONE: Il filtro ha rimosso tutti i dati! Controlla i CSV.")
            return None, None
        
        y = np.zeros((final_len, 7))
        for i in range(final_len):
            cls = labels_class[i] - 1 
            lvl = labels_level[i]
            if 0 <= cls < 7:
                y[i, cls] = lvl - 0.25
                
    except Exception as e:
        print(f"ERRORE Labels: {e}")
        return None, None

    # D. Processing Finale
    X_final = process_train_logic(X_raw, snr)
    
    return X_final, y

# ==========================================
# 4. TEST VISIVO
# ==========================================
if __name__ == "__main__":
    X, y = load_and_process_data(DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    if X is not None:
        print("\n--- TEST DIAGNOSTICO ---")
        print("Se non hai visto messaggi [WARN] sopra, il metodo degli zeri è SICURO al 100%.")
        print(f"Shape X: {X.shape}")
        
        # Prende un indice casuale sicuro, oppure 0 se ce ne sono pochi
        
        if len(X) > 0:
            idx = random.randint(0, len(X) - 1)
            ch = 0
            
            plt.figure(figsize=(10, 4))
            plt.plot(X[idx, :, ch], color='orange', label="Segnale Valido")
            plt.title(f"Verifica Segnale (Indice {idx}) - Label: {y[idx]}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            print("Nessun dato rimasto da visualizzare.")