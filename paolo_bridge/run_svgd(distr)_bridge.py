import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import tensorflow_probability as tfp
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CONFIGURAZIONE BRIDGE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")

# Percorsi Modelli Bridge
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_bridge.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_bridge.keras") 
WEIGHTS_PATH_MINMAX = os.path.join(OUTPUT_DIR, "realnvp_bridge_weights.weights.h5")

# Percorsi Parametri Normalizzazione Bridge
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_params_bridge.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor_bridge.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_svgd_bridge")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Import Loader Bridge
from data_loader_bridge import load_and_process_data, DATA_DIR_TEST, CHANNELS, SEQ_LEN, N_CLASS

# Parametri Modello
LATENT_DIM = 4
NUM_CLASSES = N_CLASS # 6 Classi
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 64
ACTIVATION = "relu"

# Parametri SVGD (Avanzato con Bandwidth)
N_PARTICLES = 200       
ITERATIONS = 1000      
LEARNING_RATE = 0.1   
LIKELIHOOD_SCALE = 12.0 
REPULSION_RATE = 0.5
STD_DEV = 0.2
BANDWITH_FACTOR = 0.1  # <--- NUOVO PARAMETRO PER ANNEALING

# IDX Selezione
IDX = 2219 # commenta per test fisso

# ==========================================
# 2. DEFINIZIONE MODELLI (Identica al Training)
# ==========================================

try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable()
class CouplingMasked(layers.Layer):
    def __init__(self, latent_dim, num_classes, reg=0.001):
        super().__init__()
        
        # 1. LABEL EMBEDDING
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
    
    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim, "num_classes": self.num_classes, 
                       "hidden_dim": self.hidden_dim, "reg": self.reg, "activation": self.activation})
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
        
        self.layers_list = [CouplingMasked(latent_dim, num_classes) for _ in range(num_coupling_layers)]

    def call(self, x, y):
        log_det_inv = 0
        z = x
        for i in range(self.num_coupling_layers):
            mask = self.masks[i]
            reversed_mask = 1 - mask
            z_masked = z * mask
            s, t = self.layers_list[i](z_masked, y)
            transformed_term = (z * reversed_mask) * tf.exp(s) + t
            z = (z * mask) + (transformed_term * reversed_mask)
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
            
            in_bounds = tf.logical_and(particles >= 0.0, particles <= 0.55)
            penalty = tf.where(in_bounds, 0.0, -1000.0 * tf.square(particles)) 
            target = (log_prob * LIKELIHOOD_SCALE) + tf.reduce_sum(penalty, axis=1)
            
        grads = tape.gradient(target, particles)
        grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        return tf.clip_by_value(grads, -100000.0, 100000.0)
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_coupling_layers": self.num_coupling_layers, "latent_dim": self.latent_dim,
                       "num_classes": self.num_classes, "hidden_dim": self.hidden_dim, "activation": self.activation})
        return config

# ==========================================
# 3. KERNEL SVGD (AGGIORNATO CON BW_FACTOR)
# ==========================================

@tf.function
def svgd_kernel_safe(theta, bw_factor): # <--- ACCETTA BW_FACTOR
    n_particles = tf.shape(theta)[0]
    theta_expand = tf.expand_dims(theta, 1)
    theta_t_expand = tf.expand_dims(theta, 0)
    pairwise_dists = tf.reduce_sum(tf.square(theta_expand - theta_t_expand), axis=2)
    
    # Median Heuristic
    h = tfp.stats.percentile(pairwise_dists, 50.0)
    h = h / tf.math.log(tf.cast(n_particles, tf.float32) + 1.0)
    
    # --- NUOVA LOGICA: Applicazione Bandwidth Factor ---
    h = tf.maximum(h * bw_factor, 0.01) 
    
    Kxy = tf.exp(-pairwise_dists / h)
    diff = theta_expand - theta_t_expand
    dx_kxy = -(2.0 / h) * tf.expand_dims(Kxy, -1) * diff
    return Kxy, dx_kxy

@tf.function
def svgd_step(particles, grads_logp, bw_factor): # <--- ACCETTA BW_FACTOR
    n_particles = tf.cast(tf.shape(particles)[0], tf.float32)
    
    # Passa il fattore dinamico al kernel
    Kxy, dx_kxy = svgd_kernel_safe(particles, bw_factor)
    
    term1 = tf.matmul(Kxy, grads_logp) 
    term2 = tf.reduce_sum(dx_kxy, axis=0) * REPULSION_RATE
    
    phi = (term1 + term2) / n_particles
    phi = tf.where(tf.math.is_nan(phi), tf.zeros_like(phi), phi)
    phi = tf.clip_by_norm(phi, 1.0)
    
    new_particles = particles + LEARNING_RATE * phi
    return new_particles

def calculate_mode_vector(particles, lower=0.0, upper=0.55, num_points=100):
    modes = []
    x_grid = np.linspace(lower, upper, num_points)
    for i in range(particles.shape[1]): 
        data = particles[:, i]
        try:
            kde = gaussian_kde(data)
            modes.append(x_grid[np.argmax(kde(x_grid))])
        except:
            modes.append(np.mean(data))
    return np.array(modes)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100 * np.mean(np.abs(y_true - y_pred) / np.where(denom==0, 1.0, denom))

def nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rng = np.max(y_true) - np.min(y_true)
    return rmse / rng if rng != 0 else 0.0

# ==========================================
# 4. MAIN & PLOTTING
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati e Controlli Bridge ---")
    X, y = load_and_process_data(DATA_DIR_TEST, is_training=False)
    
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError(f"FILE MANCANTI in {OUTPUT_DIR}.")

    z_params = np.load(PARAMS_Z_PATH)
    z_mean, z_std = z_params['mean'], z_params['std']
    
    y_scale_factor_loaded = np.load(PARAMS_Y_PATH)
    y_scale_factor_val = float(y_scale_factor_loaded.item()) if y_scale_factor_loaded.ndim == 0 else float(y_scale_factor_loaded)
    y_scale_factor_tf = tf.constant(y_scale_factor_val, dtype=tf.float32)

    try:
        keras.utils.get_custom_objects().update({"CouplingMasked": CouplingMasked, "RealNVP_Full": RealNVP_Full})
    except: pass

    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            return z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(shape=tf.shape(z_mean))

    print("--- 2. Caricamento Modelli Bridge ---")
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    try: classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)
    except: 
        print("ERRORE: Classificatore non trovato.")
        exit()

    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, activation=ACTIVATION)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    realnvp.load_weights(WEIGHTS_PATH_MINMAX)
    print("Modelli e Pesi caricati.")
    
    print(f"\n--- Analisi SVGD Bridge (Campione {IDX}) ---")

    #IDX = np.random.randint(0, len(X))

    X_sample = X[IDX:IDX+1]
    y_true = y[IDX]
    print(f"Label Vera: {y_true}")

    z_sample_raw, _, _ = encoder.predict(X_sample, verbose=0)
    z_sample_tf = tf.constant((z_sample_raw - z_mean) / z_std, dtype=tf.float32)

    y_pred_init = np.clip(classifier_cvae.predict(z_sample_raw, verbose=0)[0], 0.001, 0.54)
    print(f"Start Point (CVAE): {y_pred_init}")
    
    print(f"Generazione {N_PARTICLES} particelle...")
    particles = tf.tile(tf.expand_dims(y_pred_init, 0), [N_PARTICLES, 1])
    #particles = particles + tf.random.normal(shape=particles.shape, mean=0.0, stddev=STD_DEV)
    particles = particles + tf.random.uniform(shape=particles.shape, minval=-STD_DEV, maxval=STD_DEV)
    
    particles = tf.clip_by_value(particles, 0.0, 0.55)
    
    print(f"Esecuzione SVGD ({ITERATIONS} iterazioni) con Dynamic Bandwidth...")
    start_time = time.time()
    
    # Setup Annealing
    start_bw = 1.0
    end_bw = BANDWITH_FACTOR
    decay_steps = ITERATIONS

    for step in range(ITERATIONS):
        # --- NUOVA LOGICA: Calcolo Bandwidth Dinamico ---
        # Decrescita lineare invertita (parte da 1.0 e va verso 20.0 per "allargare" la visione o viceversa)
        # Nel codice Beam facevamo: start - (start - end) * progress
        # Se start=1 e end=20 -> va da 1 a 20. (Più alto = più repulsione/spreding)
        current_bw = start_bw - (start_bw - end_bw) * (step / decay_steps)
        current_bw_tf = tf.constant(current_bw, dtype=tf.float32)

        # Calcolo Gradienti
        log_prob_grads = realnvp.log_prob_grads(z_sample_tf, particles, y_scale_factor_tf)
        
        # Monitoring
        if step % 100 == 0:
            p_mean = tf.reduce_mean(tf.abs(log_prob_grads)).numpy()
            print(f"   [Step {step}] GradMean: {p_mean:.4f} | BW: {current_bw:.2f}")
        
        # Update con BW Factor passato
        particles = svgd_step(particles, log_prob_grads, current_bw_tf)
        particles = tf.clip_by_value(particles, 0.0, 0.55)

    elapsed = time.time() - start_time
    final_particles = particles.numpy()

    # === REPORTING ===
    mean_est = np.mean(final_particles, axis=0)
    mode_est = calculate_mode_vector(final_particles, lower=0.0, upper=0.55)
    
    y_true_2d = y_true.reshape(1, -1)
    mean_est_2d = mean_est.reshape(1, -1)
    mode_est_2d = mode_est.reshape(1, -1)
    
    metrics_mean = {
        "MAE": mean_absolute_error(y_true, mean_est),
        "MSE": mean_squared_error(y_true, mean_est),
        "RMSE": np.sqrt(mean_squared_error(y_true, mean_est)),
        "R2": r2_score(y_true, mean_est),
        "SMAPE": smape(y_true, mean_est),
        "NRMSE": nrmse(y_true, mean_est),
        "CosSim": cosine_similarity(y_true_2d, mean_est_2d)[0][0],
        "ExpVar": explained_variance_score(y_true, mean_est)
    }
    
    metrics_mode = {
        "MAE": mean_absolute_error(y_true, mode_est),
        "MSE": mean_squared_error(y_true, mode_est),
        "RMSE": np.sqrt(mean_squared_error(y_true, mode_est)),
        "R2": r2_score(y_true, mode_est),
        "SMAPE": smape(y_true, mode_est),
        "NRMSE": nrmse(y_true, mode_est),
        "CosSim": cosine_similarity(y_true_2d, mode_est_2d)[0][0],
        "ExpVar": explained_variance_score(y_true, mode_est)
    }

    print("\n" + "="*80)
    print("   REPORT STATISTICO AVANZATO (MEAN vs MODE) - Campione Singolo")
    print("="*80)
    print(f"{'Metrica':<20} | {'Media':<20} | {'Moda':<20}")
    print("-" * 66)

    keys_order = ["MAE", "MSE", "RMSE", "R2", "SMAPE", "NRMSE", "CosSim", "ExpVar"]
    for k in keys_order:
        val_mean = metrics_mean[k]
        val_mode = metrics_mode[k]
        better = ""
        if k in ["MAE", "MSE", "RMSE", "SMAPE", "NRMSE"]:
            if val_mode < val_mean: better = "<-- Mode Best"
        else:
            if val_mode > val_mean: better = "<-- Mode Best"
        print(f"{k:<20} | {val_mean:<20.5f} | {val_mode:<20.5f} {better}")
    print("-" * 66)

    std_dev_est = np.std(final_particles, axis=0)
    print("\n--- Deviazione Standard per Classe (Incertezza) ---")
    for i in range(NUM_CLASSES):
        print(f"Classe {i+1}: {std_dev_est[i]:.4f}")

    # Plotting
    print("\n--- Plotting ---")
    plt.figure(figsize=(15, 8))
    hist_color = 'limegreen'
    edge_color = 'darkgreen'
    
    for i in range(NUM_CLASSES):
        plt.subplot(2, 3, i + 1)
        sns.histplot(final_particles[:, i], kde=True, color=hist_color, edgecolor=edge_color, 
                     stat="density", bins=30, alpha=0.6)
        
        plt.axvline(y_true[i], color='red', linestyle='--', linewidth=3, label='True')
        plt.axvline(mean_est[i], color='blue', linestyle='-', linewidth=2, label='Mean')
        plt.axvline(mode_est[i], color='cyan', linestyle='-', linewidth=2, label='Mode')
        
        plt.title(f'Class {i + 1}\nTrue: {y_true[i]:.2f} | Est: {mean_est[i]:.2f}', fontsize=12)
        plt.xlabel('Damage Level')
        plt.xlim(0, 0.6) 
        if i == 0: plt.ylabel('Density')
        if i == 2: plt.legend(loc='upper right')
        
    plt.suptitle(f'SVGD Bridge Posterior - Sample {IDX}\nParticles: {N_PARTICLES} | BW Factor: {BANDWITH_FACTOR}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"svgd_bridge_distr_{IDX}.png"))
    plt.show()