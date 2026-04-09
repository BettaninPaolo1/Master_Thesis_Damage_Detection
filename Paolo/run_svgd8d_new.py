import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import tensorflow_probability as tfp
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, r2_score
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CONFIGURAZIONE (ADATTATA PER 8D)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_NAME = "models_evolution_v2" 
OUTPUT_DIR = os.path.join(SCRIPT_DIR, FOLDER_NAME)

ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae_8d.keras")
# Assicurati che questo file esista, altrimenti lo script userà un'init casuale
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_cvae_8d.keras") 
WEIGHTS_PATH_8D = os.path.join(OUTPUT_DIR, "realnvp_8d_weights.weights.h5")

# Percorsi Parametri Normalizzazione (Dal Training 8D)
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_normalization_params_8d.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor_8d.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_svgd_8d_final")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# Parametri Modello (Allineati ESATTAMENTE al Training 8D)
LATENT_DIM = 8           
NUM_CLASSES = 7
NUM_COUPLING_LAYERS = 14 
HIDDEN_DIM = 128         
ACTIVATION = "relu"      
DROPOUT_RATE = 0.1       

# Parametri SVGD
NUM_TEST_SAMPLES = 200   
N_PARTICLES = 200       
ITERATIONS = 500        
LEARNING_RATE = 0.01    
LIKELIHOOD_SCALE = 8.0  
REPULSION_RATE = 0.2
STD_DEV_INIT = 0.08   

# ==========================================
# 2. DEFINIZIONE MODELLI (Architettura 8D Split)
# ==========================================

# --- FIX COMPATIBILITÀ SERIALIZZAZIONE ---
try:
    # Keras 3 / Nuove versioni TF
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    # Keras 2 / Vecchie versioni TF
    register_serializable = keras.utils.register_keras_serializable

@register_serializable()
class CouplingMaskedSplit(layers.Layer):
    def __init__(self, latent_dim, num_classes, hidden_dim=128, reg=1e-3, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.reg = reg
        
        # LABEL EMBEDDING
        self.label_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu')
        ], name="label_embedding")

        # RAMO T (Traslazione)
        self.t_z_dense = layers.Dense(hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.t_dropout = layers.Dropout(DROPOUT_RATE) 
        self.t_l_dense = layers.Dense(hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.t_joint   = layers.Dense(hidden_dim // 2, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.t_out     = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg))

        # RAMO S (Scalatura)
        self.s_z_dense = layers.Dense(hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.s_dropout = layers.Dropout(DROPOUT_RATE) 
        self.s_l_dense = layers.Dense(hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.s_joint   = layers.Dense(hidden_dim // 2, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.s_out     = layers.Dense(latent_dim, activation="tanh", kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y, training=False):
        label_emb = self.label_net(y)

        # Calcolo T
        t_z = self.t_z_dense(x_masked)
        t_z = self.t_dropout(t_z, training=training)
        t_l = self.t_l_dense(label_emb)
        t_concat = layers.Concatenate()([t_z, t_l])
        t = self.t_joint(t_concat)
        t = self.t_out(t)

        # Calcolo S
        s_z = self.s_z_dense(x_masked)
        s_z = self.s_dropout(s_z, training=training)
        s_l = self.s_l_dense(label_emb)
        s_concat = layers.Concatenate()([s_z, s_l])
        s = self.s_joint(s_concat)
        s = self.s_out(s)

        return s, t
    
    def get_config(self):
        return super().get_config() | {
            "latent_dim": self.latent_dim, 
            "num_classes": self.num_classes, 
            "hidden_dim": self.hidden_dim,
            "reg": self.reg
        }

@register_serializable()
class RealNVP_8D_Split(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, reg=1e-3, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.reg = reg
        
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        
        # MASCHERE ALTERNATE PER 8 DIMENSIONI (Pattern 4+4 come nel training)
        mask_pattern_1 = np.array([1]*4 + [0]*4, dtype=np.float32)
        mask_pattern_2 = np.array([0]*4 + [1]*4, dtype=np.float32)
        
        masks_list = []
        for i in range(num_coupling_layers):
            if i % 2 == 0: masks_list.append(mask_pattern_1)
            else: masks_list.append(mask_pattern_2)
        self.masks = tf.constant(masks_list, dtype=tf.float32)
        
        self.layers_list = [
            CouplingMaskedSplit(latent_dim, num_classes, hidden_dim, reg, activation) 
            for _ in range(num_coupling_layers)
        ]

    def call(self, x, y, training=False):
        log_det_inv = 0
        z = x
        for i in range(self.num_coupling_layers):
            mask = self.masks[i]
            reversed_mask = 1 - mask
            
            z_masked = z * mask
            s, t = self.layers_list[i](z_masked, y, training=training)
            
            transformed = (z * reversed_mask) * tf.exp(s) + t
            
            z = (z * mask) + (transformed * reversed_mask)
            log_det_inv += tf.reduce_sum(s * reversed_mask, axis=-1)
            
        return z, log_det_inv

    @tf.function
    def log_prob_grads(self, x_fixed, particles, y_scale_factor_tf):
        # x_fixed è già normalizzato (z score)
        x_batch = tf.tile(x_fixed, [tf.shape(particles)[0], 1])
        with tf.GradientTape() as tape:
            tape.watch(particles)
            particles_scaled = particles * y_scale_factor_tf
            
            # IMPOTANTE: training=False spegne il Dropout per gradienti deterministici
            z_pred, log_det_inv = self.call(x_batch, particles_scaled, training=False)
            
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            
            # Penalità fisica [0, 0.55]
            penalty = tf.where(tf.logical_and(particles >= 0.0, particles <= 0.55), 0.0, -1000.0 * tf.square(particles)) 
            target = (log_prob * LIKELIHOOD_SCALE) + tf.reduce_sum(penalty, axis=1)
            
        grads = tape.gradient(target, particles)
        grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        return tf.clip_by_value(grads, -100.0, 100.0)
    
    def get_config(self):
        return super().get_config() | {
            "num_coupling_layers": self.num_coupling_layers, 
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim
        }

# ==========================================
# 3. KERNEL SVGD (Invariato)
# ==========================================

@tf.function
def svgd_kernel_safe(theta):
    n = tf.shape(theta)[0]
    pwd = tf.reduce_sum(tf.square(tf.expand_dims(theta, 1) - tf.expand_dims(theta, 0)), axis=2)
    h = tf.maximum(tfp.stats.percentile(pwd, 50.0) / tf.math.log(tf.cast(n, tf.float32) + 1.0), 0.01)
    Kxy = tf.exp(-pwd / h)
    dx_kxy = -(2.0 / h) * tf.expand_dims(Kxy, -1) * (tf.expand_dims(theta, 1) - tf.expand_dims(theta, 0))
    return Kxy, dx_kxy

@tf.function
def svgd_step(particles, grads_logp):
    Kxy, dx_kxy = svgd_kernel_safe(particles)
    phi = (tf.matmul(Kxy, grads_logp) + tf.reduce_sum(dx_kxy, axis=0) * REPULSION_RATE) / tf.cast(tf.shape(particles)[0], tf.float32)
    return particles + LEARNING_RATE * tf.clip_by_norm(tf.where(tf.math.is_nan(phi), 0., phi), 1.0)

# ==========================================
# 4. FUNZIONE DI VALUTAZIONE
# ==========================================
def calculate_mode_vector(particles, lower=0.0, upper=0.55):
    modes, x_grid = [], np.linspace(lower, upper, 100)
    for i in range(particles.shape[1]):
        try: modes.append(x_grid[np.argmax(gaussian_kde(particles[:, i])(x_grid))])
        except: modes.append(np.mean(particles[:, i]))
    return np.array(modes)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100 * np.mean(np.abs(y_true - y_pred) / np.where(denom==0, 1.0, denom))

def evaluate_metrics(results_df):
    print("\n" + "="*80)
    print("   REPORT DI VALUTAZIONE SVDG 8D (Internal Check)")
    print("="*80)

    def parse_col(col_name):
        val = results_df[col_name].iloc[0]
        if isinstance(val, str):
            return np.array([np.fromstring(str(x).replace('[','').replace(']','').replace('\n',' ').replace(',', ' '), sep=' ') 
                             for x in results_df[col_name]])
        return np.stack(results_df[col_name].values)

    y_true_all = parse_col('True_Vector')
    y_mean_all = parse_col('Mean_Vector')
    y_mode_all = parse_col('Mode_Vector')

    mae_mean = mean_absolute_error(y_true_all, y_mean_all)
    mae_mode = mean_absolute_error(y_true_all, y_mode_all)
    r2_mode = r2_score(y_true_all, y_mode_all)
    
    cos_sim_mean = np.mean([cosine_similarity(y_true_all[i:i+1], y_mean_all[i:i+1])[0][0] for i in range(len(y_true_all))])
    cos_sim_mode = np.mean([cosine_similarity(y_true_all[i:i+1], y_mode_all[i:i+1])[0][0] for i in range(len(y_true_all))])

    print(f"{'Metrica':<20} | {'Media (Mean Est)':<20} | {'Moda (Mode Est)':<20}")
    print("-" * 66)
    print(f"{'MAE':<20} | {mae_mean:<20.5f} | {mae_mode:<20.5f}")
    print(f"{'R2 Score':<20} | {'-':<20} | {r2_mode:<20.5f}")
    print(f"{'Cos Sim':<20} | {cos_sim_mean:<20.5f} | {cos_sim_mode:<20.5f}")
    print("-" * 66)

    true_classes = np.argmax(y_true_all, axis=1) + 1
    pred_classes = np.argmax(y_mode_all, axis=1) + 1
    acc = accuracy_score(true_classes, pred_classes)
    
    cm = confusion_matrix(true_classes, pred_classes, labels=range(1, 8))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 8), yticklabels=range(1, 8))
    plt.title(f'Confusion Matrix - SVDG 8D (Mode)\nAccuracy: {acc:.2f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_svgd_8d_mode.png"))
    plt.show()

# ==========================================
# 5. MAIN LOOP
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    # Check File Esistenza
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError(f"Parametri normalizzazione mancanti in {OUTPUT_DIR}")
    
    # Caricamento Parametri Normalizzazione 8D
    z_params = np.load(PARAMS_Z_PATH)
    z_mean, z_std = z_params['mean'], z_params['std']
    
    y_scale_factor_raw = np.load(PARAMS_Y_PATH)
    y_scale_factor_val = float(y_scale_factor_raw.item()) if y_scale_factor_raw.ndim == 0 else float(y_scale_factor_raw)
    y_scale_factor_tf = tf.constant(y_scale_factor_val, dtype=tf.float32)

    try: keras.utils.get_custom_objects().update({"CouplingMaskedSplit": CouplingMaskedSplit, "RealNVP_8D_Split": RealNVP_8D_Split})
    except: pass

    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, i): return i[0] + tf.exp(0.5 * i[1]) * tf.random.normal(tf.shape(i[0]))

    print("--- 2. Caricamento Modelli 8D ---")
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    
    # Carica Classificatore 8D (se disponibile, altrimenti errore)
    if os.path.exists(CLASSIFIER_PATH):
        print(f"Caricamento Classificatore: {CLASSIFIER_PATH}")
        classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)
    else:
        print("ATTENZIONE: Classificatore non trovato, si userà inizializzazione casuale (sconsigliato).")
        classifier_cvae = None

    # Inizializzazione RealNVP 8D
    realnvp = RealNVP_8D_Split(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM)
    
    # DUMMY PASS per costruire i pesi (Fondamentale)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    
    # Caricamento Pesi
    print(f"Caricamento pesi RealNVP da: {WEIGHTS_PATH_8D}")
    realnvp.load_weights(WEIGHTS_PATH_8D)
    
    print(f"\n[EXEC] Inizio Loop SVGD 8D su {NUM_TEST_SAMPLES} campioni...")
    np.random.seed(23)
    indices = np.random.choice(len(X), min(NUM_TEST_SAMPLES, len(X)), replace=False)
    results_list = []
    csv_file = os.path.join(RESULTS_DIR, "results_svgd_8d_batch.csv")
    
    total_start_time = time.time()
    
    for i, idx in enumerate(indices):
        iter_start = time.time() # <--- AGGIUNGI QUESTA RIGA (RIGA 1)
        try:
            X_sample = X[idx:idx+1]
            y_true = y[idx]
            true_cls = np.argmax(y_true) + 1
            
            # 1. Encoding
            z_sample_raw, _, _ = encoder.predict(X_sample, verbose=0)
            
            # 2. Z-NORMALIZATION (CRITICO PER 8D)
            z_sample_tf = tf.constant((z_sample_raw - z_mean) / z_std, dtype=tf.float32)
            
            # 3. Init Particelle
            if classifier_cvae:
                y_init = np.clip(classifier_cvae.predict(z_sample_raw, verbose=0)[0], 0.001, 0.54)
                particles = tf.tile(tf.expand_dims(y_init, 0), [N_PARTICLES, 1]) + tf.random.normal((N_PARTICLES, NUM_CLASSES), 0., STD_DEV_INIT)
            else:
                y_init = np.full((NUM_CLASSES,), 0.25) # Fallback
                particles = tf.random.uniform((N_PARTICLES, NUM_CLASSES), 0.0, 0.55)

            particles = tf.clip_by_value(particles, 0.0, 0.55)
            
            # 4. SVGD Optimization Loop
            for step in range(ITERATIONS):
                grads = realnvp.log_prob_grads(z_sample_tf, particles, y_scale_factor_tf)
                particles = svgd_step(particles, grads)
                particles = tf.clip_by_value(particles, 0.0, 0.55)

            final_p = particles.numpy()
            if np.isnan(final_p).any(): continue

            # 5. Stats & Metrics
            mean_est = np.mean(final_p, axis=0)
            mode_est = calculate_mode_vector(final_p)
            std_est = np.std(final_p, axis=0)
            
            print(f"Sample {idx:03d} | True: {true_cls} | Pred: {np.argmax(mode_est)+1} | Time: {time.time() - iter_start:.3f}s")
            y_true_2d = y_true.reshape(1, -1)
            mean_2d, mode_2d = mean_est.reshape(1, -1), mode_est.reshape(1, -1)

            row_data = {
                "Index": idx,
                "True_Class": true_cls,
                "Pred_Class_Mean": np.argmax(mean_est) + 1,
                "Pred_Class_Mode": np.argmax(mode_est) + 1,
                "True_Vector": str(y_true.tolist()),
                "Mean_Vector": str(mean_est.tolist()),
                "Mode_Vector": str(mode_est.tolist()),
                "Std_Vector": str(std_est.tolist()),
                "VAE_Init": str(y_init.tolist()),
                
                "MAE_Mean": mean_absolute_error(y_true, mean_est),
                "MSE_Mean": mean_squared_error(y_true, mean_est),
                "RMSE_Mean": np.sqrt(mean_squared_error(y_true, mean_est)),
                "R2_Mean": r2_score(y_true, mean_est),
                "CosSim_Mean": cosine_similarity(y_true_2d, mean_2d)[0][0],
                "SMAPE_Mean": smape(y_true, mean_est),
                
                "MAE_Mode": mean_absolute_error(y_true, mode_est),
                "MSE_Mode": mean_squared_error(y_true, mode_est),
                "RMSE_Mode": np.sqrt(mean_squared_error(y_true, mode_est)),
                "R2_Mode": r2_score(y_true, mode_est),
                "CosSim_Mode": cosine_similarity(y_true_2d, mode_2d)[0][0],
                "SMAPE_Mode": smape(y_true, mode_est),
            }
            results_list.append(row_data)
            
            if (i+1) % 20 == 0:
                pd.DataFrame(results_list).to_csv(csv_file, index=False)
                
        except Exception as e:
            print(f"Error on sample {idx}: {e}")

    total_time = time.time() - total_start_time
    print(f"\nTempo Totale: {total_time:.2f}s")
    
    if results_list:
        final_df = pd.DataFrame(results_list)
        final_df.to_csv(csv_file, index=False)
        print(f"\n[DONE] CSV salvato in: {csv_file}")
        evaluate_metrics(final_df)
    else:
        print("Nessun risultato valido.")