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
# 1. CONFIGURAZIONE & VARIABILE SCALE
# ==========================================
# MODIFICA QUI IL VALORE PER VEDERE L'EFFETTO SUI GRAFICI


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")

# Percorsi specifici per BEAM CASE
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_cvae.keras") 
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "realnvp_minmax14layers_weights.weights.h5")
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_normalization_params.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "diagnostics_distribution") 
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Import specifico per il BEAM
from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# Parametri Architettura
LATENT_DIM = 4
NUM_CLASSES = 7 
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 64
ACTIVATION = "relu"
DROPOUT_RATE = 0.1 

TEST_SCALE = 8.0   # <--- CAMBIA QUESTO VALORE (es. 1.0, 8.0, 50.0)

IDX = 1729

# ==========================================
# 2. DEFINIZIONE MODELLI (Con Scale Dinamico)
# ==========================================
try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@register_serializable()
class CouplingMasked(layers.Layer):
    def __init__(self, latent_dim, num_classes, reg=0.001, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.reg = reg
        
        self.label_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu')
        ])

        # Ramo T
        self.t_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_dropout = layers.Dropout(DROPOUT_RATE) 
        self.t_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.t_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_out     = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg))

        # Ramo S
        self.s_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_dropout = layers.Dropout(DROPOUT_RATE) 
        self.s_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.s_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_out     = layers.Dense(latent_dim, activation="tanh", kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y, training=False):
        label_emb = self.label_net(y)
        # Ramo T
        t_z = self.t_z_dense(x_masked)
        t_z = self.t_dropout(t_z, training=training)
        t_l = self.t_l_dense(label_emb)
        t = self.t_out(self.t_joint(layers.Concatenate()([t_z, t_l])))
        
        # Ramo S
        s_z = self.s_z_dense(x_masked)
        s_z = self.s_dropout(s_z, training=training)
        s_l = self.s_l_dense(label_emb)
        s = self.s_out(self.s_joint(layers.Concatenate()([s_z, s_l])))
        
        return s, t
    
    def get_config(self):
        return super().get_config() | {"latent_dim": self.latent_dim, "num_classes": self.num_classes, "reg": self.reg}

@register_serializable()
class RealNVP_Full(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, activation=ACTIVATION, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim))
        
        self.masks = tf.constant([[1, 1, 0, 0] if i % 2 == 0 else [0, 0, 1, 1] for i in range(num_coupling_layers)], dtype=tf.float32)
        
        self.layers_list = [CouplingMasked(latent_dim, num_classes) for _ in range(num_coupling_layers)]

    def call(self, x, y):
        log_det_inv = 0
        z = x
        for i in range(self.num_coupling_layers):
            mask = self.masks[i]
            reversed_mask = 1 - mask
            z_masked = z * mask
            
            s, t = self.layers_list[i](z_masked, y, training=False)
            
            term_to_transform = z * reversed_mask
            transformed_term = term_to_transform * tf.exp(s) + t
            z = (z * mask) + (transformed_term * reversed_mask)
            log_det_inv += tf.reduce_sum(s * reversed_mask, axis=-1)
            
        return z, log_det_inv

    # --- MODIFICA: ACCETTA likelihood_scale COME ARGOMENTO ---
    @tf.function
    def value_and_gradient(self, x, y, y_scale_factor_tf, likelihood_scale_tf):
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        likelihood_scale = tf.cast(likelihood_scale_tf, dtype=tf.float32) # Cast scala

        if len(x.shape) == 1: x = tf.expand_dims(x, 0)
        if len(y.shape) == 1: y = tf.expand_dims(y, 0)

        with tf.GradientTape() as tape:
            tape.watch(y)
            # Scaling Y (Fisico -> Modello)
            y_model = y * y_scale_factor_tf
            
            z_pred, log_det_inv = self(x, y_model, training=False)
            
            # Calcolo Likelihood
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            
            # --- APPLICAZIONE SCALA DINAMICA ---
            target = tf.reduce_sum(log_prob) * likelihood_scale
            
        grad = tape.gradient(target, y)
        if grad is None: grad = tf.zeros_like(y)
        grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
        
        return target, grad

# ==========================================
# 3. FUNZIONE DI SCANSIONE
# ==========================================

def diagnostic_scan_all_classes(model, z_fixed, y_true, y_vae, y_scale, test_scale):
    print(f"\n[DIAG] Scansione Completa Beam con LIKELIHOOD_SCALE = {test_scale}")
    
    cols = 3
    rows = int(np.ceil(NUM_CLASSES / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    # Range di scansione per il danno (Beam Case: 0.0 - 0.55)
    vals = np.linspace(0.0, 0.55, 100)
    
    # Per riferimento (gradiente iniziale dal punto VAE)
    y_vae_tf = tf.constant([y_vae], dtype=tf.float32)
    y_scale_tf = tf.constant(y_scale, dtype=tf.float32)
    test_scale_tf = tf.constant(test_scale, dtype=tf.float32)
    
    _, grad_start_tf = model.value_and_gradient(z_fixed, y_vae_tf, y_scale_tf, test_scale_tf)
    grad_start = grad_start_tf.numpy()[0]

    y_scan_base = y_vae.copy() # Si parte dal vettore VAE

    for i in range(NUM_CLASSES):
        ax = axes[i]
        log_probs = []
        grads_at_scan = []
        
        temp_y = y_scan_base.copy()
        
        for v in vals:
            temp_y[i] = v
            y_tf = tf.constant([temp_y], dtype=tf.float32)
            
            # Passiamo test_scale alla funzione
            lp, g = model.value_and_gradient(z_fixed, y_tf, y_scale_tf, test_scale_tf)
            log_probs.append(lp.numpy())
            grads_at_scan.append(g.numpy()[0][i])
            
        # Plotting Log-Prob
        ax.plot(vals, log_probs, label=f'LogProb (Scale {test_scale})', color='crimson', linewidth=2)
        
        # Plotting Gradient (su asse secondario per vedere la pendenza)
        ax2 = ax.twinx()
        ax2.plot(vals, grads_at_scan, label='Gradient', color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel("Gradient Value", color='gray')
        
        # Linee di Riferimento
        ax.axvline(y_vae[i], color='green', linestyle='--', label='VAE Start')
        ax.axvline(y_true[i], color='blue', linestyle=':', linewidth=2, label='True Val')
        
        # Annotazioni
        grad_val = grad_start[i]
        ax.set_title(f"Class {i+1} | Grad @Start: {grad_val:.2f}")
        ax.set_xlabel("Damage Level")
        if i % cols == 0: ax.set_ylabel("Weighted LogProb")
        ax.grid(True, alpha=0.3)
        
        # Legenda combinata
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if i == 0: ax.legend(lines + lines2, labels + labels2, loc='upper left')

    # Nascondi assi vuoti
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    filename = f"beam_scan_scale_{test_scale}_sample_random.png"
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path)
    print(f"\n✅ Grafico salvato in: {save_path}")
    plt.show()

# ==========================================
# 4. MAIN
# ==========================================

if __name__ == "__main__":
    # Caricamento Dati Beam
    print("Caricamento Dataset Beam...")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    # Caricamento Parametri
    z_params = np.load(PARAMS_Z_PATH)
    # Gestione sicura del caricamento scalare (a volte numpy salva come array 0-d)
    y_scale_loaded = np.load(PARAMS_Y_PATH)
    y_scale = float(y_scale_loaded) if y_scale_loaded.shape == () else float(y_scale_loaded[0])

    print(f"Scala Y caricata: {y_scale}")

    # Caricamento Modelli
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    classifier = keras.models.load_model(CLASSIFIER_PATH)
    
    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES))) # Build dummy
    realnvp.load_weights(WEIGHTS_PATH)
    print("Modelli caricati correttemente.")

    # Selezione Campione Casuale per il Test
    # IDX = np.random.randint(0, len(X))
     # Fisso per riproducibilità se vuoi confrontare scale diverse
    
    print(f"\n--- TEST SCALA SU CAMPIONE BEAM {IDX} ---")
    
    X_s = X[IDX:IDX+1]
    y_true = y[IDX]
    print(y_true)
    # Inferenza CVAE (Start Point)
    z_raw, _, _ = encoder.predict(X_s, verbose=0)
    z_norm = (z_raw - z_params['mean']) / z_params['std']
    y_vae = np.clip(classifier.predict(z_raw, verbose=0)[0], 0.001, 0.54)
    
    print(f"True Class: {np.argmax(y_true)+1}")
    print(f"VAE Pred Class: {np.argmax(y_vae)+1}")
    
    # ESECUZIONE DIAGNOSTICA
    diagnostic_scan_all_classes(realnvp, z_norm, y_true, y_vae, y_scale, TEST_SCALE)