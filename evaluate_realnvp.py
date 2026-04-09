import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 0. CONFIGURAZIONE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models_evolution")

# Paths
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder_cvae.keras")
# Nota: Usa i pesi 'onehot' se li hai riaddestrati, altrimenti quelli 'minmax'
# Se usi quelli vecchi, la logica funzionerà comunque ma l'accuratezza dipenderà dalla compatibilità
WEIGHTS_PATH = os.path.join(MODELS_DIR, "realnvp_minmax14layers_weights.weights.h5") 
PARAMS_Z_PATH = os.path.join(MODELS_DIR, "z_normalization_params.npz")

# Parametri Architettura
LATENT_DIM = 4
NUM_CLASSES = 7
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 64
ACTIVATION = "relu"
DROPOUT_RATE = 0.1 # Ininfluente in test

from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# ==========================================
# 1. DEFINIZIONE MODELLI (Copia Esatta)
# ==========================================

class CouplingMasked(layers.Layer):
    def __init__(self, latent_dim, num_classes, hidden_dim=64, reg=0.01, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.label_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu')
        ])
        self.t_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_dropout = layers.Dropout(DROPOUT_RATE)
        self.t_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.t_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_out     = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg))
        self.s_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_dropout = layers.Dropout(DROPOUT_RATE)
        self.s_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.s_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_out     = layers.Dense(latent_dim, activation="tanh", kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y, training=False):
        label_emb = self.label_net(y)
        t_z = self.t_z_dense(x_masked)
        t_l = self.t_l_dense(label_emb)
        t_concat = layers.Concatenate()([t_z, t_l])
        t = self.t_joint(t_concat)
        t = self.t_out(t)
        s_z = self.s_z_dense(x_masked)
        s_l = self.s_l_dense(label_emb)
        s_concat = layers.Concatenate()([s_z, s_l])
        s = self.s_joint(s_concat)
        s = self.s_out(s)
        return s, t

class RealNVP_BiasFix(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, reg=0.01, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        self.masks = tf.constant([[1, 1, 0, 0] if i % 2 == 0 else [0, 0, 1, 1] for i in range(num_coupling_layers)], dtype=tf.float32)
        self.layers_list = [CouplingMasked(latent_dim, num_classes, hidden_dim, reg, activation) for _ in range(num_coupling_layers)]

    def call(self, x, y):
        log_det_inv = 0
        z = x
        for i in range(self.num_coupling_layers):
            mask = self.masks[i]
            z_masked = z * mask
            s, t = self.layers_list[i](z_masked, y)
            z = (z * mask) + ((z * (1 - mask)) * tf.exp(s) + t) * (1 - mask)
            log_det_inv += tf.reduce_sum(s * (1 - mask), axis=-1)
        return z, log_det_inv

# ==========================================
# 2. LOGICA BAYESIANA VETTORIALIZZATA
# ==========================================

@tf.function
def predict_bayesian_classification(model, z_batch):
    """
    Implementa la logica 'predict_for_all_labels' del tuo esempio.
    Testa ogni campione contro tutte le 7 classi One-Hot in parallelo.
    """
    batch_size = tf.shape(z_batch)[0]
    num_classes = model.num_classes
    
    # 1. Crea matrice One-Hot Identità (7x7)
    # Questa rappresenta tutte le possibili ipotesi: [1,0..], [0,1..] etc.
    all_labels = tf.eye(num_classes) # (7, 7)
    
    # 2. Espandi Z per accoppiarlo con ogni classe
    # Da (Batch, 4) -> (Batch, 1, 4) -> Tile -> (Batch, 7, 4)
    z_expanded = tf.expand_dims(z_batch, 1)
    z_tiled = tf.tile(z_expanded, [1, num_classes, 1]) 
    
    # 3. Espandi Labels per accoppiarle con ogni campione
    # Da (7, 7) -> (1, 7, 7) -> Tile -> (Batch, 7, 7)
    labels_tiled = tf.tile(tf.expand_dims(all_labels, 0), [batch_size, 1, 1])
    
    # 4. Flattening per il Batch Processing
    # Il modello si aspetta (N_totale, dim), quindi uniamo Batch e Classi
    z_flat = tf.reshape(z_tiled, [-1, model.latent_dim]) # (Batch*7, 4)
    labels_flat = tf.reshape(labels_tiled, [-1, num_classes]) # (Batch*7, 7)
    
    # 5. Forward Pass (Tutto in un colpo solo)
    # Z -> Gaussiana Condizionata
    z_mapped, log_det = model(z_flat, labels_flat) # Output: (Batch*7, 4)
    
    # 6. Calcolo Log-Likelihood
    # P(z|y) = P_gauss(f(z)) * |det(Jacobian)|
    log_prob_base = model.distribution.log_prob(z_mapped) # (Batch*7,)
    log_likelihood = log_prob_base + log_det # (Batch*7,)
    
    # 7. Reshape per tornare a (Batch, 7)
    log_likelihood_matrix = tf.reshape(log_likelihood, [batch_size, num_classes])
    
    return log_likelihood_matrix

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati ---")
    X_test, y_test = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    # Determina Ground Truth
    if y_test.ndim > 1:
        y_true_indices = np.argmax(y_test, axis=1)
    else:
        # Se sono scalari, convertiamo in indici interi (assumendo 0..6)
        y_true_indices = np.round(y_test * (NUM_CLASSES-1)).astype(int)

    # Parametri Z
    z_params = np.load(PARAMS_Z_PATH)
    z_mean, z_std = z_params['mean'], z_params['std']

    # Carica Encoder
    class Sampling(layers.Layer):
        def call(self, i): return i[0] + tf.exp(0.5 * i[1]) * tf.random.normal(tf.shape(i[0]))
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    
    print("\n--- 2. Caricamento RealNVP ---")
    realnvp = RealNVP_BiasFix(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM)
    
    # *** FIX IMPORTANTE ***
    # Inizializziamo con shape (1, 7) per dire al modello che l'input label è un vettore 7-dim
    # Questo risolve l'errore "expected 2 variables, received 0"
    dummy_z = tf.zeros((1, LATENT_DIM))
    dummy_y = tf.zeros((1, NUM_CLASSES))
    _ = realnvp(dummy_z, dummy_y) 
    
    realnvp.load_weights(WEIGHTS_PATH)
    print("Pesi caricati correttamente!")

    print("\n--- 3. Encoding Z ---")
    z_raw, _, _ = encoder.predict(X_test, batch_size=256, verbose=1)
    z_norm = (z_raw - z_mean) / z_std
    z_tf = tf.constant(z_norm, dtype=tf.float32)
    
    print("\n--- 4. Classificazione Bayesiana (Parallelizzata) ---")
    start_time = time.time()
    
    BATCH_SIZE = 128 # Più basso perché moltiplichiamo x7 la memoria interna
    all_preds = []
    
    # Eseguiamo a batch per non saturare la GPU/RAM
    num_samples = len(z_tf)
    for i in range(0, num_samples, BATCH_SIZE):
        end = min(i + BATCH_SIZE, num_samples)
        z_batch = z_tf[i:end]
        
        # Chiama la funzione vettorializzata
        log_likelihoods = predict_bayesian_classification(realnvp, z_batch)
        
        # Argmax per trovare la classe vincente
        # axis=1 significa "tra le 7 classi per ogni riga"
        preds_batch = tf.argmax(log_likelihoods, axis=1).numpy()
        all_preds.extend(preds_batch)
        
    elapsed = time.time() - start_time
    print(f"Classificazione completata in {elapsed:.2f}s")

    # ==========================================
    # 5. RISULTATI
    # ==========================================
    pred_indices = np.array(all_preds)
    
    # Calcolo Metriche
    acc = accuracy_score(y_true_indices, pred_indices)
    cm = confusion_matrix(y_true_indices, pred_indices)
    
    print(f"\n[RISULTATI FINALI]")
    print(f"ACCURACY: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_true_indices, pred_indices, target_names=[f"Class {i+1}" for i in range(NUM_CLASSES)]))
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(1, 8), yticklabels=range(1, 8))
    plt.title(f"Confusion Matrix (Bayesian Classifier)\nAccuracy: {acc:.2%}")
    plt.xlabel("Predicted Class (Max Likelihood)")
    plt.ylabel("True Class")
    
    save_path = os.path.join(MODELS_DIR, "bayesian_classification_results.png")
    plt.savefig(save_path)
    print(f"Matrice salvata in: {save_path}")
    plt.show()