import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import tensorflow_probability as tfp
import seaborn as sns

# ==========================================
# 1. CONFIGURAZIONE (8D / Deep FiLM 3-Layers)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Cambiato MODEL_DIR per puntare alla cartella del nuovo training
MODEL_DIR = os.path.join(SCRIPT_DIR, "models_evolution_v2_deepfilm_3layers")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "models_evolution_v2", "encoder_cvae_8d.keras")
CLASSIFIER_PATH = os.path.join(SCRIPT_DIR, "models_evolution_v2", "classifier_cvae_8d.keras")

# Cambiato il nome del file pesi per matchare il nuovo training
WEIGHTS_PATH = os.path.join(MODEL_DIR, "realnvp_8d_3layer_film_weights.weights.h5")
PARAMS_Z_PATH = os.path.join(MODEL_DIR, "z_normalization_params_8d.npz")
PARAMS_Y_PATH = os.path.join(MODEL_DIR, "y_scale_factor_8d.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_svgd_distr_8d_3L")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# Parametri Architettura 8D (Devono matchare il nuovo training)
LATENT_DIM = 8          
NUM_CLASSES = 7
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 128        
ACTIVATION = "relu"
REGULARIZATION = 1e-4 # Aggiunto per FiLM

# Parametri SVGD (Inferenza)
N_PARTICLES = 150       
ITERATIONS = 500      
LEARNING_RATE = 0.01    
LIKELIHOOD_SCALE = 8.0 
REPULSION_RATE = 1
STD_DEV_INIT = 0.04 

# ==========================================
# 2. DEFINIZIONE MODELLI (DEEP FiLM 3-LAYERS)
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
        config = super().get_config()
        config.update({"units": self.units, "activation": self.activation_str, "reg": self.reg})
        return config

@register_serializable()
class CouplingMaskedDeepFiLM_3L(layers.Layer): # <--- MODIFICATO A 3L
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

        # Ramo T (3 Layers)
        self.t_layer1 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.t_layer2 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.t_layer3 = FiLMDense(hidden_dim, activation='relu', reg=reg) # <--- Nuova riga
        self.t_out = layers.Dense(latent_dim, activation='linear', 
                                  kernel_initializer='zeros', 
                                  kernel_regularizer=regularizers.l2(reg))

        # Ramo S (3 Layers)
        self.s_layer1 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.s_layer2 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.s_layer3 = FiLMDense(hidden_dim, activation='relu', reg=reg) # <--- Nuova riga
        self.s_out = layers.Dense(latent_dim, activation='tanh', 
                                  kernel_initializer='zeros',
                                  kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y):
        y_emb = self.label_embedding(y)

        # Ramo T con Residual Connections
        t = self.t_layer1(x_masked, y_emb)
        t_res1 = self.t_layer2(t, y_emb)
        t = t + t_res1
        t_res2 = self.t_layer3(t, y_emb) # <--- Nuova riga
        t = t + t_res2                  # <--- Nuova riga
        t_final = self.t_out(t)

        # Ramo S con Residual Connections
        s = self.s_layer1(x_masked, y_emb)
        s_res1 = self.s_layer2(s, y_emb)
        s = s + s_res1
        s_res2 = self.s_layer3(s, y_emb) # <--- Nuova riga
        s = s + s_res2                  # <--- Nuova riga
        s_final = self.s_out(s)

        return s_final, t_final

@register_serializable()
class RealNVP_8D(keras.Model):
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
        
        # Ora usa la classe a 3 Layer
        self.layers_list = [
            CouplingMaskedDeepFiLM_3L(latent_dim, num_classes, hidden_dim) 
            for _ in range(num_coupling_layers)
        ]

    def call(self, x, y, training=False): # Aggiunto training flag come nel training script
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
            in_bounds = tf.logical_and(particles >= 0.0, particles <= 0.55)
            penalty = tf.where(in_bounds, 0.0, -1000.0 * tf.square(particles))
            target = (log_prob * LIKELIHOOD_SCALE) + tf.reduce_sum(penalty, axis=1)
        grads = tape.gradient(target, particles)
        grads = tf.where(tf.math.is_nan(grads), tf.zeros_like(grads), grads)
        return tf.clip_by_value(grads, -1000000.0, 1000000.0)

# ==========================================
# (Da qui in poi il tuo script originale continua esattamente uguale)
# ==========================================

@tf.function
def svgd_kernel_safe(theta):
    n_particles = tf.shape(theta)[0]
    theta_expand = tf.expand_dims(theta, 1)
    theta_t_expand = tf.expand_dims(theta, 0)
    pairwise_dists = tf.reduce_sum(tf.square(theta_expand - theta_t_expand), axis=2)
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
    return particles + LEARNING_RATE * phi

if __name__ == "__main__":
    print("--- 1. Caricamento Dati 8D ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError("FILE 8D MANCANTI. Controlla models_evolution_v2_deepfilm_3layers.")

    z_params = np.load(PARAMS_Z_PATH)
    z_mean, z_std = z_params['mean'], z_params['std']
    y_scale_factor_val = float(np.load(PARAMS_Y_PATH))
    y_scale_factor_tf = tf.constant(y_scale_factor_val, dtype=tf.float32)

    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            return z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(shape=tf.shape(z_mean))

    print("--- 2. Caricamento Modelli 8D ---")
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)

    realnvp = RealNVP_8D(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    realnvp.load_weights(WEIGHTS_PATH)
    
    IDX = np.random.randint(0, len(X))
    IDX = 2797
    #IDX = 1457

    X_sample, y_true = X[IDX:IDX+1], y[IDX]
    z_raw, _, _ = encoder.predict(X_sample, verbose=0)
    z_tf = tf.constant((z_raw - z_mean) / z_std, dtype=tf.float32)
    y_init = np.clip(classifier_cvae.predict(z_raw, verbose=0)[0], 0.001, 0.54)
    print(f"Start Point (CVAE): {y_init}")
    print(f"True Point : {y_true}")
    
    particles = tf.clip_by_value(
        tf.tile(tf.expand_dims(y_init, 0), [N_PARTICLES, 1]) + 
        tf.random.normal([N_PARTICLES, NUM_CLASSES], stddev=STD_DEV_INIT), 
        0.0, 0.55
    )

    print(f"Esecuzione SVGD ({ITERATIONS} iterazioni)...")
    start_time = time.time()
    for step in range(ITERATIONS):
        log_prob_grads = realnvp.log_prob_grads(z_tf, particles, y_scale_factor_tf)
        if step % 100 == 0:
            x_batch = tf.tile(z_tf, [tf.shape(particles)[0], 1])
            z_pred, log_det_inv = realnvp(x_batch, particles * y_scale_factor_tf)
            p_mean = tf.reduce_mean(realnvp.distribution.log_prob(z_pred) + log_det_inv).numpy()
            print(f"   [Step {step}] LogProb: {p_mean:.2f}")
        particles = svgd_step(particles, log_prob_grads)
        particles = tf.clip_by_value(particles, 0.0, 0.55)

    elapsed = time.time() - start_time
    final_particles = particles.numpy()
    std_dev_est = np.std(final_particles, axis=0)
    mean_est = np.mean(final_particles, axis=0)

    # Plotting (Identico a come lo avevi nel tuo file)
    plt.figure(figsize=(15, 10))
    for i in range(NUM_CLASSES):
        plt.subplot(2, 4, i + 1)
        sns.histplot(final_particles[:, i], kde=True, color='limegreen', edgecolor='darkgreen', stat="density", bins=30, alpha=0.6)
        plt.axvline(y_true[i], color='red', linestyle='--', linewidth=3, label='True')
        plt.axvline(mean_est[i], color='blue', linestyle='-', linewidth=2, label='Mean')
        plt.title(f'Class {i + 1}\nTrue: {y_true[i]:.2f} | Est: {mean_est[i]:.2f}', fontsize=12)
        plt.xlabel('Damage Level'); plt.xlim(0, 0.6)
        if i == 6: plt.legend()
        
    plt.suptitle(f'SVGD 8D (3-Layer Deep FiLM) Posterior - Sample {IDX}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"svgd_distr_8d_{IDX}.png"))
    plt.show()