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
# 1. CONFIGURAZIONE
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")

ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_bridge.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_bridge.keras") 
# AGGIORNATO: Nome del file dei pesi dell'ultimo training
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "realnvp_bridge_3layer_film_weights.weights.h5")
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_params_bridge.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor_bridge.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "diagnostics_natural") 
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Import Loader Dati (Assicurati che il file esista nella stessa cartella)
from data_loader_bridge import load_and_process_data, DATA_DIR_TEST, CHANNELS, SEQ_LEN, N_CLASS

# AGGIORNATO: Parametri Architettura dell'ultimo training
LATENT_DIM = 4          
NUM_CLASSES = N_CLASS   
NUM_COUPLING_LAYERS = 14     
HIDDEN_DIM = 128            
REGULARIZATION = 1e-5

LIKELIHOOD_SCALE = 2.0

# Indice del campione da testare
IDX = 433

# ==========================================
# 2. DEFINIZIONE MODELLI (AGGIORNATI)
# ==========================================
try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        # Supporta sia il formato list/tuple (del nuovo training) sia i due argomenti
        if isinstance(inputs, (list, tuple)):
            z_mean, z_log_var = inputs[0], inputs[1]
        else:
            z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@register_serializable()
class FiLMDense(layers.Layer):
    """Blocco denso con modulazione FiLM."""
    def __init__(self, units, activation='relu', reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.reg = reg
        self.activation_name = activation
        
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
        return super().get_config() | {"units": self.units, "activation": self.activation_name, "reg": self.reg}

@register_serializable()
class CouplingMaskedDeepFiLM_3L(layers.Layer):
    def __init__(self, latent_dim, num_classes, hidden_dim=128, reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.reg = reg
        
        # Embedding Globale - RIGOROSAMENTE LINEARE
        self.label_embedding = keras.Sequential([
            layers.Dense(64, activation='swish', kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(128, activation='swish', kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(128, activation='linear', kernel_regularizer=regularizers.l2(reg)) 
        ], name="global_label_emb")

        # --- RAMO T (Traslazione) - 3 LAYERS ---
        self.t_layer1 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.t_layer2 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.t_layer3 = FiLMDense(hidden_dim, activation='relu', reg=reg) 
        self.t_out = layers.Dense(latent_dim, activation='linear', 
                                  kernel_initializer='zeros', 
                                  kernel_regularizer=regularizers.l2(reg))

        # --- RAMO S (Scalatura) - 3 LAYERS ---
        self.s_layer1 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.s_layer2 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.s_layer3 = FiLMDense(hidden_dim, activation='relu', reg=reg) 
        self.s_out = layers.Dense(latent_dim, activation='tanh', 
                                  kernel_initializer='zeros',
                                  kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y):
        y_emb = self.label_embedding(y)

        # --- RAMO T ---
        t = self.t_layer1(x_masked, y_emb)      
        t_res1 = self.t_layer2(t, y_emb)        
        t = t + t_res1                          
        t_res2 = self.t_layer3(t, y_emb)        
        t = t + t_res2                          
        t_final = self.t_out(t)

        # --- RAMO S ---
        s = self.s_layer1(x_masked, y_emb)      
        s_res1 = self.s_layer2(s, y_emb)        
        s = s + s_res1                          
        s_res2 = self.s_layer3(s, y_emb)        
        s = s + s_res2                          
        s_final = self.s_out(s)

        return s_final, t_final

    def get_config(self):
        return super().get_config() | {"latent_dim": self.latent_dim, "hidden_dim": self.hidden_dim, "reg": self.reg}

@register_serializable()
class RealNVP_Bridge_3L(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.reg = reg
        
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        
        # Maschere per spazio 4D (metà 1 e metà 0 alternate)
        masks = []
        for i in range(num_coupling_layers):
            mask = np.zeros(latent_dim, dtype=np.float32)
            if i % 2 == 0: mask[:latent_dim//2] = 1
            else: mask[latent_dim//2:] = 1
            masks.append(mask)
        self.masks = tf.constant(masks, dtype=tf.float32)
        
        self.layers_list = [
            CouplingMaskedDeepFiLM_3L(latent_dim, num_classes, hidden_dim, reg) 
            for _ in range(num_coupling_layers)
        ]

    def call(self, x, y):
        log_det_inv = 0
        z = x
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
    def value_and_gradient_natural(self, x, y, y_scale_factor_tf):
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        
        if len(x.shape) == 1: x = tf.expand_dims(x, 0)
        if len(y.shape) == 1: y = tf.expand_dims(y, 0)

        with tf.GradientTape() as tape:
            tape.watch(y)
            # Scaling Y (Fisico -> Modello)
            y_model = y * y_scale_factor_tf
            
            z_pred, log_det_inv = self(x, y_model)
            
            # Calcolo Likelihood Pura
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            target = tf.reduce_sum(log_prob) 
            target = tf.reduce_sum(log_prob) * LIKELIHOOD_SCALE
            
        grad = tape.gradient(target, y)
        if grad is None: grad = tf.zeros_like(y)
        grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
        
        return target, grad

    def get_config(self):
        return super().get_config() | {
            "num_coupling_layers": self.num_coupling_layers, "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim, "reg": self.reg
        }

# ==========================================
# 3. FUNZIONE DI SCANSIONE
# ==========================================

# ==========================================
# 3. FUNZIONE DI SCANSIONE (AGGIORNATA)
# ==========================================

def diagnostic_scan_natural(model, z_fixed, y_true, y_vae, y_scale):
    print(f"\n[DIAG] Scansione Completa (Scala Naturale)")
    
    cols = 3
    rows = int(np.ceil(NUM_CLASSES / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    vals = np.linspace(0.0, 0.55, 100)
    y_scale_tf = tf.constant(y_scale, dtype=tf.float32)
    
    # Gradiente Iniziale
    y_vae_tf = tf.constant([y_vae], dtype=tf.float32)
    _, grad_start = model.value_and_gradient_natural(z_fixed, y_vae_tf, y_scale_tf)
    grad_start = grad_start.numpy()[0]

    for i in range(NUM_CLASSES):
        if i >= len(axes): break
        ax = axes[i]
        log_probs = []
        
        # Copia di base per modificare solo la classe i
        base_y = y_vae.copy()
        
        for v in vals:
            base_y[i] = v
            y_tf = tf.constant([base_y], dtype=tf.float32)
            
            # Calcolo valore
            lp, _ = model.value_and_gradient_natural(z_fixed, y_tf, y_scale_tf)
            log_probs.append(lp.numpy())
            
        # Trova il punto di MASSIMA likelihood empirica in questa scansione
        max_idx = np.argmax(log_probs)
        max_val_x = vals[max_idx]
        max_val_y = log_probs[max_idx]
            
        # Plotting della curva
        ax.plot(vals, log_probs, label='Natural LogProb', color='darkblue', linewidth=2)
        
        # --- NUOVO: Evidenzia il punto di massimo ---
        # Aggiunge una stella dorata sul picco e una linea verticale
        ax.plot(max_val_x, max_val_y, marker='*', markersize=15, color='gold', markeredgecolor='black', zorder=5)
        ax.axvline(max_val_x, color='orange', linestyle='-', linewidth=2, alpha=0.8, label=f'Max Likelihood')
        
        # Riferimenti Verticali
        ax.axvline(y_vae[i], color='green', linestyle='--', label='VAE Start')
        ax.axvline(y_true[i], color='crimson', linestyle=':', linewidth=2, label='True Val')
        
        # Annotazioni: aggiunto il valore esatto del massimo nel titolo
        grad_val = grad_start[i]
        ax.set_title(f"Class {i+1} | Max LP @ {max_val_x:.3f} | Grad: {grad_val:.2f}")
        ax.set_xlabel("Damage")
        if i % cols == 0: ax.set_ylabel("LogProb (Natural)")
        ax.grid(True, alpha=0.3)
        if i == 0: 
            # Sposta la legenda in una posizione che dia meno fastidio
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2) 

    # Nascondi assi vuoti
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    filename = f"scan_natural_sample_{IDX}.png"
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path)
    print(f"\n✅ Grafico salvato in: {save_path}")
    plt.show()

# ==========================================
# 4. MAIN
# ==========================================

if __name__ == "__main__":
    # A. Caricamento Dati e Parametri
    X, y = load_and_process_data(DATA_DIR_TEST, is_training=False)
    z_params = np.load(PARAMS_Z_PATH)
    y_scale = float(np.load(PARAMS_Y_PATH))

    # B. Caricamento Modelli
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    classifier = keras.models.load_model(CLASSIFIER_PATH)
    
    # Init RealNVP_Bridge_3L (Nuova architettura)
    realnvp = RealNVP_Bridge_3L(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, REGULARIZATION)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES))) # Build
    realnvp.load_weights(WEIGHTS_PATH)
    print("✅ Modelli caricati.")

    # C. Selezione Campione
    print(f"\n--- TEST SCALA NATURALE SU CAMPIONE {IDX} ---")
    
    X_s = X[IDX:IDX+1]
    y_true = y[IDX]
    
    # D. Inferenza Preliminare
    z_raw, _, _ = encoder.predict(X_s, verbose=0)
    z_norm = (z_raw - z_params['mean']) / z_params['std']
    y_vae = np.clip(classifier.predict(z_raw, verbose=0)[0], 0.001, 0.54)
    
    print(f"True Class: {np.argmax(y_true)}")
    print(f"VAE Start:  {y_vae}")
    
    # E. Esecuzione Diagnostica
    diagnostic_scan_natural(realnvp, z_norm, y_true, y_vae, y_scale)