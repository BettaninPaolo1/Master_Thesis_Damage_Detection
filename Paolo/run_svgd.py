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

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_cvae.keras") 
# ATTENZIONE: Qui carichiamo i pesi della versione POTENZIATA (14 Layers)
WEIGHTS_PATH_MINMAX = os.path.join(OUTPUT_DIR, "realnvp_minmax14layers_weights.weights.h5")

# Percorsi Parametri Normalizzazione
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_normalization_params.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_svgd_batch")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# Parametri Modello (Aggiornati)
LATENT_DIM = 4
NUM_CLASSES = 7
NUM_COUPLING_LAYERS = 14  # <--- Aggiornato a 14 per matchare il training corretto
HIDDEN_DIM = 64
ACTIVATION = "relu"       

# Parametri SVGD
NUM_TEST_SAMPLES = 200   
N_PARTICLES = 200       
ITERATIONS = 500        
LEARNING_RATE = 0.01    
LIKELIHOOD_SCALE = 1.0  # <--- Consiglio: Alzato a 50 per picchi più netti
REPULSION_RATE = 0.5
STD_DEV_INIT = 0.1     # <--- Consiglio: 0.05 per localizzare meglio attorno al CVAE

# ==========================================
# 2. DEFINIZIONE MODELLI (Nuova Architettura)
# ==========================================

try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable()
class CouplingMasked(layers.Layer):
    def __init__(self, latent_dim, num_classes, reg=0.001, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.reg = reg
        
        # 1. LABEL EMBEDDING (Identico al file di training)
        self.label_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu')
        ])

        # 2. RAMO T (Traslazione)
        self.t_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.t_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_out     = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg))

        # 3. RAMO S (Scalatura)
        self.s_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.s_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_out     = layers.Dense(latent_dim, activation="tanh", kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y):
        # A. Calcolo Embedding
        label_emb = self.label_net(y)

        # B. Calcolo T
        t_z = self.t_z_dense(x_masked)
        t_l = self.t_l_dense(label_emb)
        t_concat = layers.Concatenate()([t_z, t_l])
        t = self.t_joint(t_concat)
        t = self.t_out(t)

        # C. Calcolo S
        s_z = self.s_z_dense(x_masked)
        s_l = self.s_l_dense(label_emb)
        s_concat = layers.Concatenate()([s_z, s_l])
        s = self.s_joint(s_concat)
        s = self.s_out(s)

        return s, t
    
    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim, "num_classes": self.num_classes, "reg": self.reg})
        return config

@register_serializable()
class RealNVP_Full(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, activation=ACTIVATION, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        mask_pattern_1 = [1, 1, 0, 0]
        mask_pattern_2 = [0, 0, 1, 1]
        masks_list = []
        for i in range(num_coupling_layers):
            if i % 2 == 0: masks_list.append(mask_pattern_1)
            else: masks_list.append(mask_pattern_2)
        self.masks = tf.constant(masks_list, dtype=tf.float32)
        
        # --- Modifica Cruciale: Chiamata senza hidden_dim/activation ---
        self.layers_list = [CouplingMasked(latent_dim, num_classes) for _ in range(num_coupling_layers)]

    def call(self, x, y):
        log_det_inv = 0
        z = x
        for i in range(self.num_coupling_layers):
            mask = self.masks[i]
            reversed_mask = 1 - mask
            z_masked = z * mask
            s, t = self.layers_list[i](z_masked, y)
            term_to_transform = z * reversed_mask
            transformed_term = term_to_transform * tf.exp(s) + t
            z = (z * mask) + (transformed_term * reversed_mask)
            log_det_inv += tf.reduce_sum(s * reversed_mask, axis=-1)
        return z, log_det_inv

    @tf.function
    def log_prob_grads(self, x_fixed, particles, y_scale_factor_tf):
        # x_fixed: già normalizzato (z_score)
        # particles: range reale (0 - 0.55)
        # y_scale_factor_tf: scalare float32
        
        x_batch = tf.tile(x_fixed, [tf.shape(particles)[0], 1])
        
        with tf.GradientTape() as tape:
            tape.watch(particles)
            
            # --- SCALING (CORRETTO) ---
            particles_scaled = particles * y_scale_factor_tf
            
            z_pred, log_det_inv = self(x_batch, particles_scaled, training=False)
            
            # Log Likelihood
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            
            # Penalty per i bordi fisici
            in_bounds = tf.logical_and(particles >= 0.0, particles <= 0.55)
            penalty = tf.where(in_bounds, 0.0, -1000.0 * tf.square(particles)) 
            
            target = (log_prob * LIKELIHOOD_SCALE) + tf.reduce_sum(penalty, axis=1)
            
        grads = tape.gradient(target, particles)
        
        # Safety clips
        grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        grads = tf.clip_by_value(grads, -100.0, 100.0) 
        return grads
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_coupling_layers": self.num_coupling_layers, "latent_dim": self.latent_dim,
                       "num_classes": self.num_classes, "hidden_dim": self.hidden_dim, "activation": self.activation})
        return config

# ==========================================
# 3. KERNEL SVGD
# ==========================================

@tf.function
def svgd_kernel_safe(theta):
    n_particles = tf.shape(theta)[0]
    theta_expand = tf.expand_dims(theta, 1)
    theta_t_expand = tf.expand_dims(theta, 0)
    pairwise_dists = tf.reduce_sum(tf.square(theta_expand - theta_t_expand), axis=2)
    
    # Median Heuristic
    h = tfp.stats.percentile(pairwise_dists, 50.0)
    h = h / tf.math.log(tf.cast(n_particles, tf.float32) + 1.0)
    h = tf.maximum(h, 0.01) 
    
    Kxy = tf.exp(-pairwise_dists / h)
    diff = theta_expand - theta_t_expand
    dx_kxy = -(2.0 / h) * tf.expand_dims(Kxy, -1) * diff
    return Kxy, dx_kxy

@tf.function
def svgd_step(particles, grads_logp):
    n_particles = tf.cast(tf.shape(particles)[0], tf.float32)
    Kxy, dx_kxy = svgd_kernel_safe(particles)
    
    term1 = tf.matmul(Kxy, grads_logp) 
    term2 = tf.reduce_sum(dx_kxy, axis=0) * REPULSION_RATE
    
    phi = (term1 + term2) / n_particles
    phi = tf.where(tf.math.is_nan(phi), tf.zeros_like(phi), phi)
    phi = tf.clip_by_norm(phi, 1.0)
    
    new_particles = particles + LEARNING_RATE * phi
    return new_particles

# ==========================================
# 4. FUNZIONE DI VALUTAZIONE
# ==========================================

def evaluate_and_plot(results_df):
    """Genera report completo con Regression Plots e Analisi dello Shift per SVGD."""
    print("\n" + "="*50)
    print("   REPORT DI VALUTAZIONE SVGD (BATCH)")
    print("="*50)

    # Parsing Robusto
    def parse_val(x):
        if isinstance(x, str): 
            return np.fromstring(x.strip('[]'), sep=' ')
        return x

    y_true_all = np.stack([parse_val(x) for x in results_df['True_Label'].values])
    y_pred_all = np.stack([parse_val(x) for x in results_df['Pred_Mean'].values])
    y_std_all  = np.stack([parse_val(x) for x in results_df['Pred_Std'].values])
    y_vae_all  = np.stack([parse_val(x) for x in results_df['VAE_Init'].values])

    # Estrazione Metriche
    true_levels = np.max(y_true_all, axis=1)
    pred_levels = y_pred_all[np.arange(len(y_pred_all)), np.argmax(y_pred_all, axis=1)]
    vae_levels = y_vae_all[np.arange(len(y_vae_all)), np.argmax(y_vae_all, axis=1)]
    
    true_classes = np.argmax(y_true_all, axis=1) + 1
    pred_classes = np.argmax(y_pred_all, axis=1) + 1

    acc = accuracy_score(true_classes, pred_classes)
    mae = mean_absolute_error(true_levels, pred_levels)
    r2 = r2_score(true_levels, pred_levels)
    mean_std = np.mean(y_std_all)

    print(f"Campioni Totali: {len(results_df)}")
    print(f"ACCURACY Classificazione: {acc:.4f} ({acc*100:.2f}%)")
    print(f"MAE Livello Danno:        {mae:.4f}")
    print(f"R2 Score (Regressione):   {r2:.4f}")
    print(f"Incertezza Media (Std):   {mean_std:.4f}")

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(true_classes, pred_classes, labels=range(1, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.title(f'SVGD Confusion Matrix\nAcc: {acc:.2f} | MAE: {mae:.3f}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    # Plot 2: Regression Analysis
    plt.figure(figsize=(8, 8))
    plt.scatter(true_levels, pred_levels, alpha=0.6, color='orange', edgecolors='k', s=60, label='SVGD Prediction')
    plt.plot([0, 0.6], [0, 0.6], 'r--', linewidth=2, label='Perfect Fit')
    plt.title(f'SVGD Regression Analysis\nR2 Score: {r2:.3f}', fontsize=14)
    plt.xlabel('True Damage Level')
    plt.ylabel('Predicted Damage Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 0.6)
    plt.ylim(0, 0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "regression_plot.png"))
    plt.close()

    # Plot 3: Shift Analysis
    plt.figure(figsize=(10, 8))
    plt.scatter(true_levels, vae_levels, marker='x', color='gray', s=50, label='VAE Initial Guess', alpha=0.5)
    plt.scatter(true_levels, pred_levels, marker='o', color='orange', s=50, label='SVGD Refined', edgecolors='k')
    
    count_shifts = 0
    for i in range(len(true_levels)):
        if abs(pred_levels[i] - vae_levels[i]) > 0.005:
            plt.plot([true_levels[i], true_levels[i]], [vae_levels[i], pred_levels[i]], color='gray', alpha=0.3)
            count_shifts += 1
            
    plt.plot([0, 0.6], [0, 0.6], 'r--', label='Ideal')
    plt.title(f'Impact of SVGD (Corrections: {count_shifts}/{len(true_levels)})', fontsize=14)
    plt.xlabel('True Damage Level')
    plt.ylabel('Estimated Damage Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 0.6)
    plt.ylim(0, 0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "vae_vs_svgd_shift.png"))
    plt.close()

    # Plot 4: Uncertainty Dist
    plt.figure(figsize=(8, 5))
    plt.hist(y_std_all.flatten(), bins=30, color='orange', edgecolor='black', alpha=0.7)
    plt.title(f'SVGD Uncertainty Distribution\nMean Std: {mean_std:.4f}')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "uncertainty_dist.png"))
    plt.close()

    print(f"Tutti i grafici salvati in: {RESULTS_DIR}")

# ==========================================
# 5. MAIN LOOP
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    # Check Parametri
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError("Parametri normalizzazione mancanti!")

    z_params = np.load(PARAMS_Z_PATH)
    z_mean, z_std = z_params['mean'], z_params['std']
    y_scale_factor_raw = np.load(PARAMS_Y_PATH)
    
    if isinstance(y_scale_factor_raw, np.ndarray):
        y_scale_factor_val = float(y_scale_factor_raw.item()) if y_scale_factor_raw.ndim == 0 else float(y_scale_factor_raw[0])
    else:
        y_scale_factor_val = float(y_scale_factor_raw)
        
    y_scale_factor_tf = tf.constant(y_scale_factor_val, dtype=tf.float32)
    print(f"Parametri Caricati: Z_Mean {z_mean.shape}, Y Scale {y_scale_factor_val:.4f}")

    # Gestione Custom Objects
    try:
        keras.utils.get_custom_objects().update({
            "CouplingMasked": CouplingMasked,
            "RealNVP_Full": RealNVP_Full
        })
    except: pass

    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    print("--- 2. Caricamento Modelli ---")
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    try:
        classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)
    except:
        print("ERRORE: Classificatore non trovato.")
        exit()

    # Inizializza RealNVP
    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, activation=ACTIVATION)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    realnvp.load_weights(WEIGHTS_PATH_MINMAX)
    
    # --- 3. LOOP SVGD BATCH ---
    print(f"\nInizio Loop SVGD su {NUM_TEST_SAMPLES} campioni...")
    indices = np.random.choice(len(X), NUM_TEST_SAMPLES, replace=False)
    results_list = []
    csv_file = os.path.join(RESULTS_DIR, "results_svgd_batch.csv")
    
    total_start_time = time.time()
    
    for i, idx in enumerate(indices):
        iter_start = time.time()
        try:
            X_sample = X[idx:idx+1]
            y_true = y[idx]
            true_class = np.argmax(y_true) + 1
            
            # 1. Encoding & Z-Norm
            z_sample_raw, _, _ = encoder.predict(X_sample, verbose=0)
            z_sample_norm = (z_sample_raw - z_mean) / z_std
            z_sample_tf = tf.constant(z_sample_norm, dtype=tf.float32)
            
            # 2. VAE Init
            y_pred_init = classifier_cvae.predict(z_sample_raw, verbose=0)[0]
            y_pred_init = np.clip(y_pred_init, 0.001, 0.54)
            
            # 3. Particelle Init
            particles = tf.tile(tf.expand_dims(y_pred_init, 0), [N_PARTICLES, 1])
            noise = tf.random.normal(shape=particles.shape, mean=0.0, stddev=STD_DEV_INIT)
            particles = particles + noise
            particles = tf.clip_by_value(particles, 0.0, 0.55)
            
            # 4. SVGD Optimization Loop
            for step in range(ITERATIONS):
                log_prob_grads = realnvp.log_prob_grads(z_sample_tf, particles, y_scale_factor_tf)
                particles = svgd_step(particles, log_prob_grads)
                particles = tf.clip_by_value(particles, 0.0, 0.55)

            final_particles = particles.numpy()
            
            if np.isnan(final_particles).any():
                print(f" [Sample {i+1}] NaN Detected. Skipping.")
                continue

            # 5. Stats
            mean_est = np.mean(final_particles, axis=0)
            std_est = np.std(final_particles, axis=0)
            pred_class = np.argmax(mean_est) + 1
            avg_std = np.mean(std_est)
            
            print(f"[Sample {i+1}/{NUM_TEST_SAMPLES}] IDX: {idx} | "
                  f"True: {true_class} | Pred: {pred_class} | "
                  f"StdDev: {avg_std:.4f} | Time: {time.time()-iter_start:.2f}s")
            
            results_list.append({
                "Index": idx,
                "True_Label": y_true,
                "Pred_Mean": mean_est,
                "Pred_Std": std_est,
                "VAE_Init": y_pred_init
            })
            
            if (i+1) % 10 == 0:
                pd.DataFrame(results_list).to_csv(csv_file, index=False)
                
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            continue

    total_time = time.time() - total_start_time
    print(f"\nTempo Totale: {total_time:.2f}s")
    
    if len(results_list) > 0:
        final_df = pd.DataFrame(results_list)
        final_df.to_csv(csv_file, index=False)
        evaluate_and_plot(final_df)
    else:
        print("Nessun risultato valido.")