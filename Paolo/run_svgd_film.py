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
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_cvae.keras") 

# ATTENZIONE: Nome dei pesi aggiornato al modello LITE
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "realnvp_lite_film_weights.weights.h5")

# Percorsi Parametri Normalizzazione
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_normalization_params.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor.npy")

# Nuova cartella risultati
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_svgd502_litefilm_relu")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# Parametri Modello (Aggiornati all'ultimo addestramento LITE)
LATENT_DIM = 4          
NUM_CLASSES = 7         
NUM_COUPLING_LAYERS = 4      # <--- 4 per la versione LITE
HIDDEN_DIM = 32              # <--- 32 per la versione LITE
ACTIVATION = 'relu'          # <--- Deve corrispondere

# Parametri SVGD
NUM_TEST_SAMPLES = 500
N_PARTICLES = 100       
ITERATIONS = 300        
LEARNING_RATE = 0.01 
LIKELIHOOD_SCALE = 8.0  
REPULSION_RATE = 0.1
STD_DEV_INIT = 0.05
BANDWITH_FACTOR = 20

# ==========================================
# 2. DEFINIZIONE MODELLI (LITE FiLM)
# ==========================================

try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable()
class FiLMDense(layers.Layer):
    """Layer FiLM Singolo, mantenuto liscio con LayerNorm."""
    def __init__(self, units, activation='relu', reg=1e-1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = keras.activations.get(activation)
        self.reg = reg

        self.dense = layers.Dense(units, kernel_regularizer=regularizers.l2(reg))
        self.ln = layers.LayerNormalization(epsilon=1e-5) 
        
        self.gamma_dense = layers.Dense(units, kernel_regularizer=regularizers.l2(reg), kernel_initializer="zeros")
        self.beta_dense  = layers.Dense(units, kernel_regularizer=regularizers.l2(reg), kernel_initializer="zeros")

    def call(self, x, cond):
        h = self.dense(x)
        h = self.ln(h)               
        
        gamma = self.gamma_dense(cond)
        beta = self.beta_dense(cond)
        
        h_film = (1.0 + gamma) * h + beta  
        if self.activation is not None:
            h_film = self.activation(h_film)
        return h_film

    def get_config(self):
        return super().get_config() | {"units": self.units, "activation": self.activation_name, "reg": self.reg}

@register_serializable()
class CouplingMaskedLiteFiLM(layers.Layer): 
    """Versione Lite: Niente Residuals, 1 solo passaggio FiLM."""
    def __init__(self, latent_dim, num_classes, hidden_dim=32, reg=1e-1, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.reg = reg
        
        self.label_embedding = keras.Sequential([
            layers.Dense(32, activation=ACTIVATION),
            layers.Dense(32, activation=ACTIVATION) 
        ], name="global_label_emb")

        # Ramo T (1 Layer Singolo)
        self.t_layer = FiLMDense(hidden_dim, activation=ACTIVATION, reg=reg)
        self.t_out = layers.Dense(latent_dim, activation='linear', 
                                  kernel_initializer='zeros', 
                                  kernel_regularizer=regularizers.l2(reg))

        # Ramo S (1 Layer Singolo)
        self.s_layer = FiLMDense(hidden_dim, activation=ACTIVATION, reg=reg)
        self.s_out = layers.Dense(latent_dim, activation='tanh', 
                                  kernel_initializer='zeros',
                                  kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y, training=False):
        y_emb = self.label_embedding(y)

        # Ramo T (Semplice feedforward)
        t = self.t_layer(x_masked, y_emb)
        t_final = self.t_out(t)

        # Ramo S (Semplice feedforward)
        s = self.s_layer(x_masked, y_emb)
        s_final = self.s_out(s)

        return s_final, t_final
        
    def get_config(self):
        return super().get_config() | {"latent_dim": self.latent_dim, "num_classes": self.num_classes, 
                                       "hidden_dim": self.hidden_dim, "reg": self.reg}

@register_serializable()
class RealNVP_LiteFiLM(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim=32, reg=1e-1, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        
        self.masks = tf.constant(
            [[1, 1, 0, 0] if i % 2 == 0 else [0, 0, 1, 1] for i in range(num_coupling_layers)], 
            dtype=tf.float32
        )
        
        self.layers_list = [
            CouplingMaskedLiteFiLM(latent_dim, num_classes, hidden_dim, reg) 
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
            
            term_to_transform = z * reversed_mask
            transformed_term = term_to_transform * tf.exp(s) + t
            
            z = (z * mask) + (transformed_term * reversed_mask)
            log_det_inv += tf.reduce_sum(s * reversed_mask, axis=-1)
            
        return z, log_det_inv

    # Integriamo la funzione SVGD per calcolare il gradiente sulle particelle
    @tf.function
    def log_prob_grads(self, x_fixed, particles, y_scale_factor_tf):
        x_batch = tf.tile(x_fixed, [tf.shape(particles)[0], 1])
        with tf.GradientTape() as tape:
            tape.watch(particles)
            particles_scaled = particles * y_scale_factor_tf
            z_pred, log_det_inv = self(x_batch, particles_scaled, training=False)
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            
            # Penalità fisica [0, 0.55]
            penalty = tf.where(tf.logical_and(particles >= 0.0, particles <= 0.55), 0.0, -1000.0 * tf.square(particles)) 
            target = (log_prob * LIKELIHOOD_SCALE) + tf.reduce_sum(penalty, axis=1)
            
        grads = tape.gradient(target, particles)
        grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        return tf.clip_by_value(grads, -100.0, 100.0)
    
    def get_config(self):
        return super().get_config() | {"num_coupling_layers": self.num_coupling_layers, "latent_dim": self.latent_dim}

# ==========================================
# 3. KERNEL SVGD
# ==========================================

@tf.function
def svgd_kernel_safe(theta, bw_factor):
    n_particles = tf.shape(theta)[0]
    theta_expand = tf.expand_dims(theta, 1)
    theta_t_expand = tf.expand_dims(theta, 0)
    pairwise_dists = tf.reduce_sum(tf.square(theta_expand - theta_t_expand), axis=2)
    
    h = tfp.stats.percentile(pairwise_dists, 50.0)
    h = h / tf.math.log(tf.cast(n_particles, tf.float32) + 1.0)
    h = tf.maximum(h * bw_factor, 0.01) 
    
    Kxy = tf.exp(-pairwise_dists / h)
    diff = theta_expand - theta_t_expand
    dx_kxy = -(2.0 / h) * tf.expand_dims(Kxy, -1) * diff
    return Kxy, dx_kxy

@tf.function
def svgd_step(particles, grads_logp, bw_factor):
    n_particles = tf.cast(tf.shape(particles)[0], tf.float32)
    Kxy, dx_kxy = svgd_kernel_safe(particles, bw_factor)
    
    term1 = tf.matmul(Kxy, grads_logp) 
    term2 = tf.reduce_sum(dx_kxy, axis=0) * REPULSION_RATE
    
    phi = (term1 + term2) / n_particles
    phi = tf.where(tf.math.is_nan(phi), tf.zeros_like(phi), phi)
    phi = tf.clip_by_norm(phi, 1.0)
    
    new_particles = particles + LEARNING_RATE * phi
    return new_particles

# ==========================================
# 4. FUNZIONI DI VALUTAZIONE
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
    print("   REPORT DI VALUTAZIONE SVDG AVANZATO (Internal Check)")
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
    plt.title(f'Confusion Matrix - SVDG \nAccuracy: {acc:.2f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_svgd_mode.png"))
    plt.close() 

# ==========================================
# 5. MAIN LOOP
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError("Parametri normalizzazione mancanti!")

    z_params = np.load(PARAMS_Z_PATH)
    z_mean, z_std = z_params['mean'], z_params['std']
    y_scale_factor_raw = np.load(PARAMS_Y_PATH)
    y_scale_factor_val = float(y_scale_factor_raw.item()) if y_scale_factor_raw.ndim == 0 else float(y_scale_factor_raw)
    y_scale_factor_tf = tf.constant(y_scale_factor_val, dtype=tf.float32)

    try: 
        keras.utils.get_custom_objects().update({
            "FiLMDense": FiLMDense,
            "CouplingMaskedLiteFiLM": CouplingMaskedLiteFiLM, 
            "RealNVP_LiteFiLM": RealNVP_LiteFiLM
        })
    except: pass

    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, i): return i[0] + tf.exp(0.5 * i[1]) * tf.random.normal(tf.shape(i[0]))

    print("--- 2. Caricamento Modelli ---")
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)

    # Inizializziamo il nuovo modello LITE FiLM
    realnvp = RealNVP_LiteFiLM(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    realnvp.load_weights(WEIGHTS_PATH)
    print("✅ Pesi RealNVP LITE FiLM caricati correttamente.")
    
    print(f"\n[EXEC] Inizio Loop SVGD su {NUM_TEST_SAMPLES} campioni...")
    
    np.random.seed(67)
    indices = np.random.choice(len(X), min(NUM_TEST_SAMPLES, len(X)), replace=False)
    results_list = []
    csv_file = os.path.join(RESULTS_DIR, "results_svgd_batch.csv")
    
    total_start_time = time.time()
    
    for i, idx in enumerate(indices):
        start_t = time.time()
        try:
            X_sample = X[idx:idx+1]
            y_true = y[idx]
            true_cls = np.argmax(y_true) + 1
            
            # 1. Encoding & Z-Norm
            z_sample_raw, _, _ = encoder.predict(X_sample, verbose=0)
            z_sample_tf = tf.constant((z_sample_raw - z_mean) / z_std, dtype=tf.float32)
            
            # 2. VAE Init
            y_init = np.clip(classifier_cvae.predict(z_sample_raw, verbose=0)[0], 0.001, 0.54)
            
            # 3. Particelle Init
            particles = tf.tile(tf.expand_dims(y_init, 0), [N_PARTICLES, 1]) + tf.random.normal((N_PARTICLES, NUM_CLASSES), 0., STD_DEV_INIT)
            particles = tf.clip_by_value(particles, 0.0, 0.55)

            diff = tf.abs(particles - tf.tile(tf.expand_dims(y_init, 0), [N_PARTICLES, 1]))
            movement = tf.reduce_mean(diff).numpy()
            
            start_bw = 1.0
            end_bw = BANDWITH_FACTOR
            decay_steps = ITERATIONS
            
            # 4. SVGD Optimization Loop
            for step in range(ITERATIONS):
                current_bw = start_bw - (start_bw - end_bw) * (step / decay_steps)
                current_bw_tf = tf.constant(current_bw, dtype=tf.float32)

                particles = svgd_step(particles, 
                                      realnvp.log_prob_grads(z_sample_tf, particles, y_scale_factor_tf),
                                      current_bw_tf)
                particles = tf.clip_by_value(particles, 0.0, 0.55)

            final_p = particles.numpy()
            if np.isnan(final_p).any(): continue

            # 5. Stats & Metrics
            mean_est = np.mean(final_p, axis=0)
            mode_est = calculate_mode_vector(final_p)
            std_est = np.std(final_p, axis=0)
            
            y_true_2d = y_true.reshape(1, -1)
            mean_2d, mode_2d = mean_est.reshape(1, -1), mode_est.reshape(1, -1)
            elapsed = time.time() - start_t

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
                "Time": elapsed,
                
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

            pred_cls = np.argmax(mean_est) + 1
            print(f"{i+1:04d} Done: IDX = {idx} | ({elapsed:.1f}s) | True: {true_cls} | Pred: {pred_cls} | Movement: {movement:.5f} | Std_dev: {np.mean(std_est):.4f}")
            
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