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
# 1. CONFIGURAZIONE & PERCORSI
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution") 
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_svgd_stats8")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# File specifici (DEVONO PUNTARE ALLA CARTELLA DOVE HAI ADDESTRATO)
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_cvae.keras") 
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "realnvp_overfit_weights.weights.h5")

# Statistiche salvate dal training
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_normalization_params.npz") 
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor.npy")

from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# Parametri Architettura (DEVONO MATCHARE PERFETTAMENTE IL TRAINING)
LATENT_DIM = 4
NUM_CLASSES = 7
NUM_COUPLING_LAYERS = 14   
HIDDEN_DIM = 64
ACTIVATION = "relu"      
REGULARIZATION = 1e-4

# Parametri SVGD (Senza Ancora)
NUM_TEST_SAMPLES = 100    
N_PARTICLES = 100       
ITERATIONS = 500          
LEARNING_RATE = 0.01      
LIKELIHOOD_SCALE = 8.0    
REPULSION_RATE = 0.1
STD_DEV_INIT = 0.05        
BANDWITH_FACTOR = 0.5     

# ==========================================
# 2. DEFINIZIONE MODELLI (Esatta Copia del Training)
# ==========================================

try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable()
class FiLMDense(layers.Layer):
    def __init__(self, units, activation='relu', reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = keras.activations.get(activation)
        self.reg = reg

        self.dense = layers.Dense(units, kernel_regularizer=regularizers.l2(reg))
        self.ln = layers.LayerNormalization(epsilon=1e-5) 
        self.gamma_dense = layers.Dense(units, kernel_initializer="zeros", kernel_regularizer=regularizers.l2(reg))
        self.beta_dense  = layers.Dense(units, kernel_initializer="zeros", kernel_regularizer=regularizers.l2(reg))

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
class CouplingMaskedDeepFiLM_3L(layers.Layer): 
    def __init__(self, latent_dim, num_classes, hidden_dim=64, reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.reg = reg
        
        self.label_embedding = keras.Sequential([
            layers.Dense(hidden_dim, activation='swish', kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(hidden_dim, activation='swish', kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(hidden_dim, activation='linear', kernel_regularizer=regularizers.l2(reg)) 
        ], name="global_label_emb")

        self.t_layer1 = FiLMDense(hidden_dim, activation=ACTIVATION, reg=reg)
        self.t_layer2 = FiLMDense(hidden_dim, activation=ACTIVATION, reg=reg)
        self.t_layer3 = FiLMDense(hidden_dim, activation=ACTIVATION, reg=reg) 
        self.t_out = layers.Dense(latent_dim, activation='linear', kernel_initializer='zeros', kernel_regularizer=regularizers.l2(reg))

        self.s_layer1 = FiLMDense(hidden_dim, activation=ACTIVATION, reg=reg)
        self.s_layer2 = FiLMDense(hidden_dim, activation=ACTIVATION, reg=reg)
        self.s_layer3 = FiLMDense(hidden_dim, activation=ACTIVATION, reg=reg) 
        self.s_out = layers.Dense(latent_dim, activation='tanh', kernel_initializer='zeros', kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y, training=False):
        y_emb = self.label_embedding(y)
        t = self.t_layer1(x_masked, y_emb)
        t = t + self.t_layer2(t, y_emb)
        t = t + self.t_layer3(t, y_emb) 
        t_final = self.t_out(t)

        s = self.s_layer1(x_masked, y_emb)
        s = s + self.s_layer2(s, y_emb)
        s = s + self.s_layer3(s, y_emb) 
        s_final = self.s_out(s)
        return s_final, t_final
        
    def get_config(self):
        return super().get_config() | {"latent_dim": self.latent_dim, "num_classes": self.num_classes, "hidden_dim": self.hidden_dim, "reg": self.reg}

@register_serializable()
class RealNVP_Overfit(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim=64, reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.reg = reg
        
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        
        self.masks = tf.constant(
            [[1, 1, 0, 0] if i % 2 == 0 else [0, 0, 1, 1] for i in range(num_coupling_layers)], 
            dtype=tf.float32
        )
        
        self.layers_list = [
            CouplingMaskedDeepFiLM_3L(latent_dim, num_classes, hidden_dim, reg) 
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

    @tf.function
    # Rimosso y_init_batch dalla firma
    def log_prob_grads(self, x_fixed, particles, y_scale_factor_tf): 
        x_batch = tf.tile(x_fixed, [tf.shape(particles)[0], 1])
        
        with tf.GradientTape() as tape:
            tape.watch(particles)
            particles_scaled = particles * y_scale_factor_tf
            z_pred, log_det_inv = self(x_batch, particles_scaled, training=False)
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            
            # Limiti Fisici [0.0, 0.6]
            in_bounds = tf.logical_and(particles >= 0.0, particles <= 0.6)
            penalty = tf.where(in_bounds, 0.0, -1000.0 * tf.square(particles))
            
            # Ancora rimossa. Il target è solo Likelihood + Penalty sui limiti
            target = (log_prob * LIKELIHOOD_SCALE) + tf.reduce_sum(penalty, axis=1)
            
        grads = tape.gradient(target, particles)
        grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        return tf.clip_by_value(grads, -1000.0, 1000.0)
    
    def get_config(self):
        return super().get_config() | {
            "num_coupling_layers": self.num_coupling_layers, 
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "reg": self.reg
        }

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
    h = tf.maximum(h * bw_factor, 0.1) 
    
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
    phi = tf.clip_by_norm(phi, 0.5) 
    
    new_particles = particles + LEARNING_RATE * phi
    return new_particles

# ==========================================
# 4. VALUTAZIONE
# ==========================================
def calculate_mode_vector(particles, lower=0.0, upper=0.6):
    modes, x_grid = [], np.linspace(lower, upper, 100)
    for i in range(particles.shape[1]):
        try: 
            kde = gaussian_kde(particles[:, i])
            kde.covariance_factor = lambda: .25
            kde._compute_covariance()
            modes.append(x_grid[np.argmax(kde(x_grid))])
        except: 
            modes.append(np.mean(particles[:, i]))
    return np.array(modes)

def calculate_mean_vector(particles):
    if hasattr(particles, 'numpy'): particles = particles.numpy()
    return np.mean(particles, axis=0)

# ==========================================
# 5. MAIN LOOP
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    # Caricamento Statistiche Media/Std (Z Normalizzato)
    if not os.path.exists(PARAMS_Z_PATH):
        raise FileNotFoundError(f"Statistiche Z non trovate in: {PARAMS_Z_PATH}")

    z_stats = np.load(PARAMS_Z_PATH)
    z_mean, z_std = z_stats['mean'], z_stats['std']
    
    y_scale_factor_val = float(np.load(PARAMS_Y_PATH))
    y_scale_factor_tf = tf.constant(y_scale_factor_val, dtype=tf.float32)
    
    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, i): return i[0] + tf.exp(0.5 * i[1]) * tf.random.normal(tf.shape(i[0]))

    print("--- 2. Caricamento Modelli ---")
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)

    print("--- 3. Inizializzazione RealNVP ---")
    realnvp = RealNVP_Overfit(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, REGULARIZATION)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    
    print(f"Caricamento pesi da: {WEIGHTS_PATH}")
    realnvp.load_weights(WEIGHTS_PATH)
    
    print(f"\n[EXEC] Inizio Loop SVGD su {NUM_TEST_SAMPLES} campioni...")
    np.random.seed(67) 
    indices = np.random.choice(len(X), min(NUM_TEST_SAMPLES, len(X)), replace=False)
    results_list = []
    csv_file = os.path.join(RESULTS_DIR, "results_svgd_batch.csv")
    
    for i, idx in enumerate(indices):
        start_t = time.time()
        X_sample = X[idx:idx+1]
        y_true = y[idx]
        true_cls = np.argmax(y_true) + 1
        
        # 1. Encoding & Norm Standard (Mean/Std)
        z_sample_raw, _, _ = encoder.predict(X_sample, verbose=0)
        z_sample_tf = tf.constant((z_sample_raw - z_mean) / z_std, dtype=tf.float32)
        
        # 2. Init Particelle (CVAE Prior)
        y_init = np.clip(classifier_cvae.predict(z_sample_raw, verbose=0)[0], 0.001, 0.549)
        y_init_batch = tf.tile(tf.expand_dims(y_init, 0), [N_PARTICLES, 1])
        
        particles = tf.clip_by_value(
            y_init_batch + tf.random.normal([N_PARTICLES, NUM_CLASSES], stddev=STD_DEV_INIT), 
            0.0, 0.55
        )
        
        # 3. SVGD Loop
        for step in range(ITERATIONS):
            current_bw = BANDWITH_FACTOR
            particles = svgd_step(
                particles, 
                # Rimosso il passaggio di y_init_batch
                realnvp.log_prob_grads(z_sample_tf, particles, y_scale_factor_tf),
                tf.constant(current_bw, dtype=tf.float32)
            )
            particles = tf.clip_by_value(particles, 0.0, 0.55)

        final_p = particles.numpy()
        
        # 4. Statistiche
        mode_est = calculate_mode_vector(final_p)
        mean_est = calculate_mean_vector(final_p)
        std_est = np.std(final_p, axis=0)
        elapsed = time.time() - start_t

        pred_cls_mean = np.argmax(mean_est) + 1

        row_data = {
            "Index": idx,
            "True_Class": true_cls,
            "Pred_Class_Mean": pred_cls_mean,
            "Pred_Class_Mode": np.argmax(mode_est) + 1,
            "Time_Seconds": elapsed,
            "True_Vector": str(y_true.tolist()),
            "Mean_Vector": str(mean_est.tolist()),
            "Mode_Vector": str(mode_est.tolist()),
            "Std_Vector": str(std_est.tolist()),
            "VAE_Init": str(y_init.tolist())
        }
        results_list.append(row_data)
        
        if (i+1) % 10 == 0:
            pd.DataFrame(results_list).to_csv(csv_file, index=False)
      
        print(f"{i+1:04d} Done | IDX: {idx} | True: {true_cls} | Pred (Mean): {pred_cls_mean} | Time: {elapsed:.2f}s | Unc: {np.mean(std_est):.4f}")

    if results_list:
        final_df = pd.DataFrame(results_list)
        final_df.to_csv(csv_file, index=False)
        print(f"\n[DONE] CSV salvato in: {csv_file}")