import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import tensorflow_probability as tfp
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
import arviz as az
import seaborn as sns

# ==========================================
# 1. CONFIGURAZIONE NUTS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_cvae.keras") 
WEIGHTS_PATH_MINMAX= os.path.join(OUTPUT_DIR, "realnvp_minmax14layers_weights.weights.h5")

# PERCORSI PARAMETRI DI NORMALIZZAZIONE
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_normalization_params.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_nuts_corrected") 
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR, TEST_DIR

# Configurazione Parametri
LATENT_DIM = 4
NUM_CLASSES = 7
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 64
ACTIVATION = "relu" # Deve matchare il training

# Tuning NUTS
DRAWS = 1000       
TUNE = 500        
CHAINS = 2         
LIKELIHOOD_SCALE = 4.0 # Riportato a 1.0 (standard Bayesiano), alzalo solo se la posterior è troppo larga
TARGET_ACCEPT = 0.80   # Alzato per gestire meglio la geometria complessa

# ==========================================
# 2. DEFINIZIONE MODELLO (BRIDGE TF-PYMC)
# ==========================================

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

        # Calcolo T
        t_concat = layers.Concatenate()([self.t_z_dense(x_masked), self.t_l_dense(label_emb)])
        t = self.t_out(self.t_joint(t_concat))

        # Calcolo S
        s_concat = layers.Concatenate()([self.s_z_dense(x_masked), self.s_l_dense(label_emb)])
        s = self.s_out(self.s_joint(s_concat))
        return s, t

class RealNVP_Full(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, activation=ACTIVATION):
        super().__init__()
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        mask_pattern_1 = [1, 1, 0, 0]
        mask_pattern_2 = [0, 0, 1, 1]
        self.masks = tf.constant(
            [mask_pattern_1 if i % 2 == 0 else mask_pattern_2 for i in range(num_coupling_layers)], 
            dtype=tf.float32
        )
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
    def value_and_gradient(self, x, y, y_scale_factor_tf):
        # [MODIFICA 2] Casting esplicito per sicurezza (Bridge 32-64 bit in ingresso)
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        
        if len(x.shape) == 1: x = tf.expand_dims(x, 0)
        if len(y.shape) == 1: y = tf.expand_dims(y, 0)

        with tf.GradientTape() as tape:
            tape.watch(y)
            
            # [MODIFICA 4] Scaling Dinamico della Y
            # NUTS propone y "fisico" (es. 0.3), il modello vuole y "scalato" (es. 0.3 * factor)
            y_scaled = y * y_scale_factor_tf
            
            z_pred, log_det_inv = self(x, y_scaled, training=False)
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            target = tf.reduce_sum(log_prob) * LIKELIHOOD_SCALE

        grad = tape.gradient(target, y)
        
        # [MODIFICA 1] Il "Salvagente" sui Gradienti (NaN Protection)
        # Se il gradiente esplode o è NaN, lo forziamo a zero per non far crashare NUTS
        grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
        
        return target, grad

class RealNVPGradientOp(Op):
    def __init__(self, model, x_fixed, y_scale_factor):
        self.model = model
        self.x_fixed = x_fixed 
        self.y_scale_factor = tf.constant(y_scale_factor, dtype=tf.float32)
        
    def make_node(self, y):
        y = pt.as_tensor_variable(y)
        return Apply(self, [y], [y.type()])
        
    def perform(self, node, inputs, outputs):
        y_val = inputs[0]
        _, grad_val = self.model.value_and_gradient(self.x_fixed, y_val, self.y_scale_factor)
        # [MODIFICA 2] Bridge 32-64 bit in uscita
        outputs[0][0] = grad_val.numpy().astype(np.float64).flatten()

class RealNVPLLogpOp(Op):
    def __init__(self, model, x_fixed, y_scale_factor):
        self.model = model
        self.x_fixed = x_fixed
        self.y_scale_factor = tf.constant(y_scale_factor, dtype=tf.float32)
        self.grad_op = RealNVPGradientOp(model, x_fixed, y_scale_factor)
        
    def make_node(self, y):
        y = pt.as_tensor_variable(y)
        return Apply(self, [y], [pt.dscalar()]) 
        
    def perform(self, node, inputs, outputs):
        y_val = inputs[0]
        logp_val, _ = self.model.value_and_gradient(self.x_fixed, y_val, self.y_scale_factor)
        # [MODIFICA 2] Bridge 32-64 bit in uscita
        outputs[0][0] = np.array(logp_val.numpy(), dtype=np.float64)
        
    def grad(self, inputs, output_gradients):
        y = inputs[0]
        return [output_gradients[0] * self.grad_op(y)]

# ==========================================
# 3. ESECUZIONE MAIN
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati e Parametri ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    # Check File Esistenza
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError("Mancano i file di normalizzazione. Esegui training.")

    # Carica parametri
    z_params = np.load(PARAMS_Z_PATH)
    z_mean = z_params['mean']
    z_std = z_params['std']
    y_scale_factor = np.load(PARAMS_Y_PATH) # Scalare (float)
    
    print(f"Norm Params: Z_mean shape {z_mean.shape}, Y_scale {y_scale_factor:.4f}")

    # Dummy Sampling per caricare Encoder
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            return z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(shape=(batch, dim))

    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)

    # Inizializza RealNVP
    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, activation=ACTIVATION)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES))) # Build
    realnvp.load_weights(WEIGHTS_PATH_MINMAX)
    
    # Selezione Campione
    IDX = 2237
    print(f"\n--- Analisi MCMC NUTS (Corrected) su Campione {IDX} ---")
    
    X_sample = X[IDX:IDX+1]
    y_true = y[IDX]
    
    # 1. Encoding RAW
    z_sample_raw, _, _ = encoder.predict(X_sample, verbose=0)
    
    # [MODIFICA 3] Normalizzazione dello Spazio Latente (CRUCIALE)
    # Il RealNVP è stato trainato su dati (z - mean) / std. Dobbiamo fare lo stesso qui.
    z_sample_norm = (z_sample_raw - z_mean) / z_std
    
    # Start Point
    y_pred_init = classifier_cvae.predict(z_sample_raw, verbose=0)[0]
    y_pred_init = np.clip(y_pred_init, 0.001, 0.54) # Clip safe per start point
    print(f"Punto di partenza (CVAE): {y_pred_init}")

    # Inizializzazione Op Custom con i dati corretti
    # Qui passiamo z_sample_norm (non raw!) e il y_scale_factor
    logp_op = RealNVPLLogpOp(realnvp, z_sample_norm, y_scale_factor)
    
    start_vals = [{'label_vector': y_pred_init} for _ in range(CHAINS)]

    print("\n[Start] Campionamento NUTS...")
    start_time = time.time()
    
    with pm.Model() as model:
        # Prior Uniforme su scala REALE (0 - 0.55)
        # NUTS esplorerà questo spazio "fisico"
        label_vector = pm.Uniform('label_vector', lower=0.0, upper=0.55, shape=(NUM_CLASSES,))
        
        # Likelihood custom (Il ponte verso TensorFlow)
        pm.Potential("likelihood", logp_op(label_vector))

        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,       
            chains=CHAINS,
            cores=1, # Importante 1 core per evitare conflitti TF/PyMC
            init='jitter+adapt_diag', # Migliore inizializzazione
            start=start_vals,  
            step=pm.NUTS(target_accept=TARGET_ACCEPT),
            discard_tuned_samples=True,
            progressbar=True
        )
        
    elapsed = time.time() - start_time
    print(f"[End] MCMC NUTS completato in {elapsed:.2f} secondi.")

    # ==========================================
    # 4. DIAGNOSTICA
    # ==========================================
    
    # Verifica divergenze
    div = trace.sample_stats["diverging"].values.sum()
    print(f"\nDivergenze Totali: {div}")
    if div > 0:
        print("WARN: Ci sono divergenze. Prova ad aumentare TARGET_ACCEPT o ridurre LABEL_NOISE_STD nel training.")

    # Plotting
    posterior_flat = trace.posterior['label_vector'].values.reshape(-1, NUM_CLASSES)
    mean_est = np.mean(posterior_flat, axis=0)
    
    plt.figure(figsize=(15, 10))
    for i in range(NUM_CLASSES):
        plt.subplot(2, 4, i + 1)
        plt.hist(posterior_flat[:, i], bins=50, alpha=0.7, color='teal', density=True)
        plt.xlim(0.0, 0.6) 
        plt.axvline(y_true[i], color='red', linestyle='--', linewidth=2, label='True')
        plt.axvline(mean_est[i], color='orange', linestyle='-', linewidth=2, label='Mean')
        plt.title(f'Class {i + 1}')
        if i == 0: plt.legend()
        
    plt.suptitle(f'NUTS MCMC Posterior (Corrected) - Sample {IDX}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"nuts_corrected_{IDX}.png"))
    plt.show()

    az.plot_trace(trace)
    plt.savefig(os.path.join(RESULTS_DIR, f"nuts_trace_{IDX}.png"))