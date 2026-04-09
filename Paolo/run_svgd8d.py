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
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import gaussian_kde

# ==========================================
# 1. CONFIGURAZIONE (3-LAYER Deep FiLM)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cartelle:
# 1. Cartella del modello RealNVP specifico (3 Layers)
MODEL_DIR = os.path.join(SCRIPT_DIR, "models_evolution_v2_deepfilm_3layers")
# 2. Cartella condivisa dove stanno Encoder e Classificatore
SHARED_DIR = os.path.join(SCRIPT_DIR, "models_evolution_v2")

# Percorsi Modelli
ENCODER_PATH = os.path.join(SHARED_DIR, "encoder_cvae_8d.keras")
CLASSIFIER_PATH = os.path.join(SHARED_DIR, "classifier_cvae_8d.keras")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "realnvp_8d_3layer_film_weights.weights.h5")

# Percorsi Parametri (Specifici del training RealNVP)
PARAMS_Z_PATH = os.path.join(MODEL_DIR, "z_normalization_params_8d.npz")
PARAMS_Y_PATH = os.path.join(MODEL_DIR, "y_scale_factor_8d.npy")

# Output
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_svgd_8d_deepfilm3")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# Parametri Architettura
LATENT_DIM = 8
NUM_CLASSES = 7
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 128        
ACTIVATION = "relu"

# Parametri SVGD
# Parametri SVGD
NUM_TEST_SAMPLES = 1000   
N_PARTICLES = 100       
ITERATIONS = 500        
LEARNING_RATE = 0.1   
LIKELIHOOD_SCALE = 8.0  
REPULSION_RATE = 0.1
STD_DEV_INIT = 0.05
BANDWITH_FACTOR = 1

# ==========================================
# 2. UTILS
# ==========================================
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100 * np.mean(np.abs(y_true - y_pred) / np.where(denom==0, 1.0, denom))

def compute_metrics_row(idx, y_true, mean_est, mode_est, y_init, std_est=None, time_taken=0.0):
    y_true_2d = y_true.reshape(1, -1)
    mean_2d = mean_est.reshape(1, -1)
    mode_2d = mode_est.reshape(1, -1)

    row_data = {
        "Index": idx,
        "True_Class": np.argmax(y_true) + 1,
        "Pred_Class_Mean": np.argmax(mean_est) + 1,
        "Pred_Class_Mode": np.argmax(mode_est) + 1,
        "Time_Seconds": time_taken,
        "True_Vector": str(y_true.tolist()),
        "Mean_Vector": str(mean_est.tolist()),
        "Mode_Vector": str(mode_est.tolist()),
        "Std_Vector": str(std_est.tolist()) if std_est is not None else "[]",
        "VAE_Init": str(y_init.tolist()),
        "MAE_Mean": mean_absolute_error(y_true, mean_est),
        "MSE_Mean": mean_squared_error(y_true, mean_est),
        "R2_Mean": r2_score(y_true, mean_est),
        "CosSim_Mean": cosine_similarity(y_true_2d, mean_2d)[0][0],
        "SMAPE_Mean": smape(y_true, mean_est),
        "MAE_Mode": mean_absolute_error(y_true, mode_est),
        "MSE_Mode": mean_squared_error(y_true, mode_est),
        "R2_Mode": r2_score(y_true, mode_est),
        "CosSim_Mode": cosine_similarity(y_true_2d, mode_2d)[0][0],
        "SMAPE_Mode": smape(y_true, mode_est),
    }
    return row_data

# ==========================================
# 3. DEFINIZIONE MODELLI (3-LAYER RESIDUAL)
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
        self.activation_str = activation
        self.reg = reg
        self.dense = layers.Dense(units, kernel_regularizer=regularizers.l2(reg))
        self.activation = layers.Activation(activation)
        self.film_gen = layers.Dense(units * 2, kernel_initializer='zeros') 

    def call(self, x, label_emb):
        out = self.dense(x)
        film_params = self.film_gen(label_emb)
        gamma, beta = tf.split(film_params, 2, axis=-1)
        out = out * (1.0 + gamma) + beta
        return self.activation(out)
    
    def get_config(self):
        return super().get_config() | {"units": self.units, "activation": self.activation_str, "reg": self.reg}

@register_serializable()
class CouplingMaskedDeepFiLM_3L(layers.Layer):
    def __init__(self, latent_dim, num_classes, hidden_dim=128, reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.reg = reg
        
        self.label_embedding = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu') 
        ], name="global_label_emb")

        # --- RAMO T (3 Layers) ---
        self.t_layer1 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.t_layer2 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.t_layer3 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.t_out = layers.Dense(latent_dim, activation='linear', 
                                  kernel_initializer='zeros', kernel_regularizer=regularizers.l2(reg))

        # --- RAMO S (3 Layers) ---
        self.s_layer1 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.s_layer2 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.s_layer3 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.s_out = layers.Dense(latent_dim, activation='tanh', 
                                  kernel_initializer='zeros', kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y):
        y_emb = self.label_embedding(y)

        # Ramo T (Residual: Layer 2 e 3)
        t = self.t_layer1(x_masked, y_emb)
        t_res1 = self.t_layer2(t, y_emb)
        t = t + t_res1
        t_res2 = self.t_layer3(t, y_emb)
        t = t + t_res2
        t_final = self.t_out(t)

        # Ramo S (Residual: Layer 2 e 3)
        s = self.s_layer1(x_masked, y_emb)
        s_res1 = self.s_layer2(s, y_emb)
        s = s + s_res1
        s_res2 = self.s_layer3(s, y_emb)
        s = s + s_res2
        s_final = self.s_out(s)

        return s_final, t_final
    
    def get_config(self):
        return super().get_config() | {"latent_dim": self.latent_dim, "num_classes": self.num_classes, 
                       "hidden_dim": self.hidden_dim, "reg": self.reg}

@register_serializable()
class RealNVP_8D_3L(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        masks = []
        for i in range(num_coupling_layers):
            mask = np.zeros(latent_dim, dtype=np.float32)
            if i % 2 == 0: mask[:latent_dim//2] = 1
            else: mask[latent_dim//2:] = 1
            masks.append(mask)
        self.masks = tf.constant(masks, dtype=tf.float32)
        
        self.layers_list = [
            CouplingMaskedDeepFiLM_3L(latent_dim, num_classes, hidden_dim) 
            for _ in range(num_coupling_layers)
        ]

    def call(self, x, y):
        log_det_inv = 0; z = x
        for i in range(self.num_coupling_layers):
            mask = self.masks[i]
            reversed_mask = 1 - mask
            z_masked = z * mask
            s, t = self.layers_list[i](z_masked, y)
            transformed = (z * reversed_mask) * tf.exp(s) + t
            z = (z * mask) + (transformed * reversed_mask)
            log_det_inv += tf.reduce_sum(s * reversed_mask, axis=-1)
        return z, log_det_inv

    @tf.function
    def log_prob_grads(self, x_fixed, particles, y_scale_factor_tf):
        x_batch = tf.tile(x_fixed, [tf.shape(particles)[0], 1])
        with tf.GradientTape() as tape:
            tape.watch(particles)
            particles_scaled = particles * y_scale_factor_tf
            z_pred, log_det_inv = self(x_batch, particles_scaled, training=False)
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            # Vincolo fisico [0, 0.55]
            in_bounds = tf.logical_and(particles >= 0.0, particles <= 0.55)
            penalty = tf.where(in_bounds, 0.0, -1000.0 * tf.square(particles))
            target = (log_prob * LIKELIHOOD_SCALE) + tf.reduce_sum(penalty, axis=1)
        grads = tape.gradient(target, particles)
        grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        return tf.clip_by_value(grads, -100.0, 100.0)
    
    def get_config(self):
        return super().get_config() | {"num_coupling_layers": self.num_coupling_layers, "latent_dim": self.latent_dim,
                       "num_classes": self.num_classes, "hidden_dim": self.hidden_dim}

# ==========================================
# 4. KERNEL SVGD
# ==========================================
@tf.function
def svgd_kernel_safe(theta, bw_factor):
    n_particles = tf.shape(theta)[0]
    theta_expand = tf.expand_dims(theta, 1)
    theta_t_expand = tf.expand_dims(theta, 0)
    pairwise_dists = tf.reduce_sum(tf.square(theta_expand - theta_t_expand), axis=2)
    
    # Median Heuristic
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

def calculate_mode_vector(particles, lower=0.0, upper=0.55):
    modes = []
    x_grid = np.linspace(lower, upper, 200)
    for i in range(particles.shape[1]):
        data = particles[:, i]
        if np.std(data) < 1e-5:
            modes.append(np.mean(data))
        else:
            try:
                jitter = np.random.normal(0, 1e-12, size=data.shape)
                kde = gaussian_kde(data + jitter)
                modes.append(x_grid[np.argmax(kde(x_grid))])
            except:
                modes.append(np.mean(data))
    return np.array(modes)

# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati 8D ---")
    X, y_true_all_data = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError(f"FILE 8D MANCANTI in {MODEL_DIR}")

    z_params = np.load(PARAMS_Z_PATH)
    z_mean, z_std = z_params['mean'], z_params['std']
    y_scale_factor_val = float(np.load(PARAMS_Y_PATH))
    y_scale_factor_tf = tf.constant(y_scale_factor_val, dtype=tf.float32)

    try:
        keras.utils.get_custom_objects().update({
            "FiLMDense": FiLMDense,
            "CouplingMaskedDeepFiLM_3L": CouplingMaskedDeepFiLM_3L,
            "RealNVP_8D_3L": RealNVP_8D_3L
        })
    except: pass

    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            return z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(shape=tf.shape(z_mean))

    print("--- 2. Caricamento Modelli 8D (3-Layer) ---")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder non trovato: {ENCODER_PATH}")
        
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    try: classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)
    except: 
        print("ERRORE: Classificatore 8D non trovato.")
        exit()

    # Inizializzazione 3-Layers
    realnvp = RealNVP_8D_3L(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    
    print(f"Caricamento Pesi: {WEIGHTS_PATH}")
    realnvp.load_weights(WEIGHTS_PATH)
    
    #np.random.seed(58) 94.5%
    #np.random.seed(67)
    np.random.seed(23)
    indices = np.random.choice(len(X), NUM_TEST_SAMPLES, replace=False)

    results_list = []
    
    print(f"\n--- Inizio SVGD 8D (3 Layers + Res) su {NUM_TEST_SAMPLES} campioni ---")
    csv_file = os.path.join(RESULTS_DIR, "results_svgd_8d_deepfilm.csv")

    for i, idx in enumerate(indices):
        iter_start = time.time()
        
        X_sample, y_true_orig = X[idx:idx+1], y_true_all_data[idx]
        z_raw, _, _ = encoder.predict(X_sample, verbose=0)
        z_tf = tf.constant((z_raw - z_mean) / z_std, dtype=tf.float32)
        
        # Init CVAE
        y_init = np.clip(classifier_cvae.predict(z_raw, verbose=0)[0], 0.001, 0.54)
        
        particles = tf.clip_by_value(
            tf.tile(tf.expand_dims(y_init, 0), [N_PARTICLES, 1]) + 
            tf.random.normal([N_PARTICLES, NUM_CLASSES], stddev=STD_DEV_INIT), 
            0.0, 0.55
        )
        
        # SVGD Loop
        start_bw = 1.0
        end_bw = BANDWITH_FACTOR
        decay_steps = ITERATIONS
        for _ in range(ITERATIONS):
            current_bw = start_bw - (start_bw - end_bw) * ( _ / decay_steps)
            current_bw_tf = tf.constant(current_bw, dtype=tf.float32)

            # PASSA current_bw_tf alla funzione
            particles = svgd_step(particles, 
                                      realnvp.log_prob_grads(z_tf, particles, y_scale_factor_tf),
                                      current_bw_tf)
            particles = tf.clip_by_value(particles, 0.0, 0.55)

        # Stats
        final_particles = particles.numpy()
        mean_est = np.mean(final_particles, axis=0)
        mode_est = calculate_mode_vector(final_particles)
        std_est = np.std(final_particles, axis=0)
        
        iter_end = time.time()
        elapsed = iter_end - iter_start

        row_data = compute_metrics_row(idx, y_true_orig, mean_est, mode_est, y_init, std_est, time_taken=elapsed)
        results_list.append(row_data)
        
        true_cls = np.argmax(y_true_orig) + 1
        pred_cls_mean = np.argmax(mean_est) + 1
        
        print(f"{i+1} Done: IDX = {idx:04d} | ({elapsed:.1f}s) | True: {true_cls} | Pred: {pred_cls_mean} | Std_dev : {np.mean(std_est):.4f} ")


        if (i+1) % 10 == 0:
            pd.DataFrame(results_list).to_csv(csv_file, index=False)

    final_df = pd.DataFrame(results_list)
    final_df.to_csv(csv_file, index=False)
    
    mae_mean = final_df["MAE_Mean"].mean()
    acc_mean = accuracy_score(final_df["True_Class"], final_df["Pred_Class_Mean"])
    avg_time = final_df["Time_Seconds"].mean()

    print(f"\n--- SVGD 8D (3L) Completato ---")
    print(f"Statistiche salvate in: {csv_file}")
    print(f"Accuracy (MEAN): {acc_mean:.4f}")
    print(f"MAE Livello (MEAN): {mae_mean:.4f}")
    print(f"Tempo Medio: {avg_time:.2f}s")