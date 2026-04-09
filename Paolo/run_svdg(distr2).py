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

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_cvae.keras") 
WEIGHTS_PATH_MINMAX= os.path.join(OUTPUT_DIR, "realnvp_minmax14layers_weights.weights.h5")

# Percorsi Parametri Normalizzazione
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_normalization_params.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_svgd_distr")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# Parametri Modello
LATENT_DIM = 4
NUM_CLASSES = 7
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 64
ACTIVATION = "relu" # Assicurati che il training sia stato fatto con TANH!

# Parametri SVGD
N_PARTICLES = 200      
ITERATIONS = 1000
LEARNING_RATE = 0.01  # Se esplode, scendi a 0.005 o 0.001
LIKELIHOOD_SCALE = 8.0 
REPULSION_RATE = 5
STD_DEV = 0.05
BANDWITH_FACTOR = 1

# ==========================================
# 2. DEFINIZIONE MODELLI
# ==========================================

try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable()
class CouplingMasked(layers.Layer):
    def __init__(self, latent_dim, num_classes, reg=0.001):
        super().__init__()
        
        # 1. LABEL EMBEDDING (Identico al file di training)
        # Questa rete deve esistere per caricare i pesi corretti
        self.label_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu')
        ])

        # 2. RAMO T (Traslazione)
        # Input Z
        self.t_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        # Input Label Embedding
        self.t_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        # Unione
        self.t_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_out     = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg))

        # 3. RAMO S (Scalatura)
        # Input Z
        self.s_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        # Input Label Embedding
        self.s_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        # Unione
        self.s_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_out     = layers.Dense(latent_dim, activation="tanh", kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y):
        # A. Calcolo Embedding
        label_emb = self.label_net(y)

        # B. Calcolo T
        t_z = self.t_z_dense(x_masked)
        t_l = self.t_l_dense(label_emb)
        # Concatenazione dei due rami processati
        t_concat = layers.Concatenate()([t_z, t_l])
        t = self.t_joint(t_concat)
        t = self.t_out(t)

        # C. Calcolo S
        s_z = self.s_z_dense(x_masked)
        s_l = self.s_l_dense(label_emb)
        # Concatenazione dei due rami processati
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
        
        # Rimuovi hidden_dim e activation dalla chiamata
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
            # La Chain Rule di TF gestirà automaticamente la derivata attraverso questa moltiplicazione.
            # grad_particles = grad_scaled * scale_factor
            particles_scaled = particles * y_scale_factor_tf
            
            z_pred, log_det_inv = self(x_batch, particles_scaled, training=False)
            
            # Log Likelihood
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            
            # Penalty per i bordi fisici (sulle particelle REALI)
            in_bounds = tf.logical_and(particles >= 0.0, particles <= 0.55)
            penalty = tf.where(in_bounds, 0.0, -1000.0 * tf.square(particles)) 
            
            target = (log_prob * LIKELIHOOD_SCALE) + tf.reduce_sum(penalty, axis=1)
            
        grads = tape.gradient(target, particles)
        
        # Safety clips
        grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        #grads = tf.clip_by_value(grads, -10.0, 10.0) 
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

# ==========================================
# 4. MAIN & PLOTTING
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati e Controlli ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    # Check File Esistenza
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError("FILE MANCANTI: z_normalization_params.npz o y_scale_factor.npy non trovati. Esegui 'train_realnvp_bias_fix.py' prima.")

    # Caricamento Parametri Normalizzazione
    z_params = np.load(PARAMS_Z_PATH)
    z_mean = z_params['mean']
    z_std = z_params['std']
    
    # Caricamento e Check Scaling Factor
    y_scale_factor_loaded = np.load(PARAMS_Y_PATH)
    # Estraiamo il valore scalare puro
    if isinstance(y_scale_factor_loaded, np.ndarray):
        y_scale_factor_val = float(y_scale_factor_loaded.item()) if y_scale_factor_loaded.ndim == 0 else float(y_scale_factor_loaded[0])
    else:
        y_scale_factor_val = float(y_scale_factor_loaded)
        
    print(f"   -> Z Mean shape: {z_mean.shape}")
    print(f"   -> Y Scale Factor: {y_scale_factor_val:.4f}")
    
    if y_scale_factor_val == 1.0 or y_scale_factor_val == 0.0:
        print("\nATTENZIONE: Scale Factor è 1.0 o 0.0. Sei sicuro che il training abbia salvato il fattore corretto?")
        print("    Se il max_val dei dati era ~0.55, il fattore dovrebbe essere ~1.8.")

    # Convertiamo in tensore costante TF
    y_scale_factor_tf = tf.constant(y_scale_factor_val, dtype=tf.float32)

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

    # Inizializza RealNVP con pesi
    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, activation=ACTIVATION)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    realnvp.load_weights(WEIGHTS_PATH_MINMAX)
    
    # IDX Selezione
    IDX = np.random.randint(0, len(X))
    #IDX = 1261 # Esempio fisso
    #IDX = 2752
    #IDX = 2955

    print(f"\n--- Analisi SVGD (Campione {IDX}) ---")
    X_sample = X[IDX:IDX+1]
    y_true = y[IDX]
    print(f"Label Vera: {y_true}")

    # Encoding e Normalizzazione Z
    z_sample_raw, _, _ = encoder.predict(X_sample, verbose=0)
    z_sample_norm = (z_sample_raw - z_mean) / z_std
    z_sample_tf = tf.constant(z_sample_norm, dtype=tf.float32)

    # Start Point
    y_pred_init = classifier_cvae.predict(z_sample_raw, verbose=0)[0]
    y_pred_init = np.clip(y_pred_init, 0.001, 0.54)
    print(f"Start Point (CVAE): {y_pred_init}")
    
    # Particelle
    print(f"Generazione {N_PARTICLES} particelle...")
    particles = tf.tile(tf.expand_dims(y_pred_init, 0), [N_PARTICLES, 1])
    particles = particles + tf.random.normal(shape=particles.shape, mean=0.0, stddev=STD_DEV)
    particles = tf.clip_by_value(particles, 0.0, 0.55)
    
    # Loop
    print(f"Esecuzione SVGD ({ITERATIONS} iterazioni)...")
    start_time = time.time()

    start_bw = 1.0
    end_bw = BANDWITH_FACTOR
    decay_steps = ITERATIONS
    
    for step in range(ITERATIONS):
        # Calcolo Gradienti con Scaling
        log_prob_grads = realnvp.log_prob_grads(z_sample_tf, particles, y_scale_factor_tf)

        current_bw = start_bw - (start_bw - end_bw) * (step / decay_steps)
        current_bw_tf = tf.constant(current_bw, dtype=tf.float32)
        
        # Monitoring
        if step % 100 == 0:
            x_batch_monitor = tf.tile(z_sample_tf, [tf.shape(particles)[0], 1])
            # Anche qui dobbiamo scalare per il forward pass di controllo!
            particles_scaled_mon = particles * y_scale_factor_tf 
            
            z_pred, log_det_inv = realnvp(x_batch_monitor, particles_scaled_mon)
            total_log_prob = realnvp.distribution.log_prob(z_pred) + log_det_inv
            
            p_mean = tf.reduce_mean(total_log_prob).numpy()
            g_mean = tf.reduce_mean(tf.abs(log_prob_grads)).numpy()
            print(f"   [Step {step}] LogProb: {p_mean:.2f} | GradMean: {g_mean:.4f}")
            
            if np.isnan(p_mean) or np.isnan(g_mean):
                print("!!! CRITICAL: NaN detected. Stopping.")
                break
        
        # Update
        particles = svgd_step(particles, log_prob_grads, current_bw_tf)
        particles = tf.clip_by_value(particles, 0.0, 0.55)

    elapsed = time.time() - start_time
    final_particles = particles.numpy()

    # --- AGGIUNTA PER CALCOLO E STAMPA DEVIAZIONE STANDARD ---
    std_dev_est = np.std(final_particles, axis=0) # Calcola STD lungo le particelle (axis=0)

    print("\n--- Deviazione Standard per Classe (Incertezza) ---")
    for i in range(NUM_CLASSES):
        print(f"Classe {i+1}: {std_dev_est[i]:.4f}")
    # ---------------------------------------------------------

    print(f"SVGD Finito in {elapsed:.2f}s")

    # 1. Identificazione della MODA (per ogni classe tramite KDE)
    mode_est = []
    x_grid = np.linspace(0.0, 0.55, 200)
    
    for i in range(NUM_CLASSES):
        # Controllo di sicurezza: se la STD è quasi zero, le particelle sono collassate.
        # Saltiamo la KDE (che andrebbe in crash) e usiamo direttamente la media.
        if std_dev_est[i] < 1e-6:
            mode_val = np.mean(final_particles[:, i])
            print(f"   -> KDE bypassata per Classe {i+1} (Varianza nulla). Moda = {mode_val:.4f}")
        else:
            try:
                kde = gaussian_kde(final_particles[:, i])
                mode_val = x_grid[np.argmax(kde(x_grid))]
            except np.linalg.LinAlgError:
                # Fallback estremo in caso di altri errori di algebra lineare
                mode_val = np.mean(final_particles[:, i])
                print(f"   -> Errore LinAlg su Classe {i+1}. Moda approssimata alla media = {mode_val:.4f}")
                
        mode_est.append(mode_val)
        
    mode_est = np.array(mode_est)

    # 2. Identificazione del PUNTO DI MASSIMA LIKELIHOOD (tra le particelle finali)
    # Ricalcoliamo la log-probabilità per le particelle ottenute
    x_batch_final = tf.tile(z_sample_tf, [tf.shape(particles)[0], 1])
    particles_scaled_final = particles * y_scale_factor_tf
    z_pred_final, log_det_inv_final = realnvp(x_batch_final, particles_scaled_final)
    total_log_prob_final = realnvp.distribution.log_prob(z_pred_final) + log_det_inv_final
    
    # Troviamo l'indice della particella con la log-probabilità più alta
    idx_max_lik = tf.argmax(total_log_prob_final).numpy()
    max_lik_point = final_particles[idx_max_lik]

    # 3. Calcolo della Distanza Euclidea
    dist_mode_lik = np.linalg.norm(mode_est - max_lik_point)
    avg_std_dev = np.mean(std_dev_est)

    print("\n" + "="*50)
    print("ANALISI DI CONVERGENZA (MODA vs MAX LIKELIHOOD)")
    print("-" * 50)
    print(f"Punto Moda (KDE):        {np.round(mode_est, 4)}")
    print(f"Punto Max Likelihood:    {np.round(max_lik_point, 4)}")
    print(f"Distanza Euclidea:       {dist_mode_lik:.6f}") 
    print(f"STD_DEV:                 {avg_std_dev:.6f}")
    
    if dist_mode_lik < 0.05:
        print("-> Le particelle sono perfettamente collassate sul massimo.")
    else:
        print("-> Esiste una dispersione o la repulsione bilancia l'attrazione.")
    print("="*50 + "\n")

    # Plotting
    print("\n--- Plotting ---")
    mean_est = np.mean(final_particles, axis=0)
    
    plt.figure(figsize=(15, 10))
    hist_color = 'limegreen'
    edge_color = 'darkgreen'
    
    for i in range(NUM_CLASSES):
        plt.subplot(2, 4, i + 1)
        sns.histplot(final_particles[:, i], kde=True, color=hist_color, edgecolor=edge_color, 
                     stat="density", bins=30, alpha=0.6)
        
        plt.axvline(y_true[i], color='red', linestyle='--', linewidth=3, label='True')
        plt.axvline(mean_est[i], color='blue', linestyle='-', linewidth=2, label='Mean')
        
        plt.title(f'Class {i + 1}\nTrue: {y_true[i]:.2f} | Est: {mean_est[i]:.2f}', fontsize=12)
        plt.xlabel('Damage Level')
        plt.xlim(0, 0.6) 
        if i == 0: plt.ylabel('Density')
        if i == 6: plt.legend()
        
    plt.suptitle(f'SVGD Posterior - Sample {IDX}\nParticles: {N_PARTICLES}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"svgd_distr_{IDX}.png"))
    plt.show()