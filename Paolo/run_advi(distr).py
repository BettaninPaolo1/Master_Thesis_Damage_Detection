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
# 1. CONFIGURAZIONE ADVI
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_cvae.keras") 

# --- MODIFICA 1: Percorso Pesi Aggiornato (14 layers) ---
WEIGHTS_PATH_MINMAX = os.path.join(OUTPUT_DIR, "realnvp_minmax14layers_weights.weights.h5")

# PERCORSI PARAMETRI NORMALIZZAZIONE
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_normalization_params.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_advi")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR, TEST_DIR

LATENT_DIM = 4
NUM_CLASSES = 7
# --- MODIFICA 2: Numero di Layers aumentato a 14 ---
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 64
ACTIVATION = "relu"

# Tuning ADVI
ITERATIONS = 10000  
LIKELIHOOD_SCALE = 8.0

# ==========================================
# 2. DEFINIZIONE MODELLO (NUOVA ARCHITETTURA)
# ==========================================

# --- MODIFICA 3: Nuova classe CouplingMasked (Embedding + T/S Branches) ---
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

# --- MODIFICA 4: Aggiornamento RealNVP_Full ---
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
            transformed = (z * reversed_mask) * tf.exp(s) + t
            z = (z * mask) + (transformed * reversed_mask)
            log_det_inv += tf.reduce_sum(s * reversed_mask, axis=-1)
        return z, log_det_inv

    @tf.function
    def value_and_gradient(self, x, y, y_scale_factor_tf):
        # 1. Casting esplicito PRIMA del calcolo
        # TensorFlow a volte perde il gradiente se si fa il cast di una variabile non watchata
        x_tf = tf.cast(x, dtype=tf.float32)
        y_tf = tf.cast(y, dtype=tf.float32) # Questa è la variabile che useremo
        
        # Gestione dimensioni
        if len(x_tf.shape) == 1:
            x_tf = tf.expand_dims(x_tf, 0)
        
        with tf.GradientTape() as tape:
            # 2. WATCH CRUCIALE: Osserviamo la versione float32
            tape.watch(y_tf)
            
            # Gestione batch Y
            y_in = y_tf
            if len(y_tf.shape) == 1:
                y_in = tf.expand_dims(y_tf, 0)
            
            # Scaling Y (Moltiplicazione element-wise)
            y_scaled = y_in * y_scale_factor_tf
            
            # Forward Pass
            z_pred, log_det_inv = self(x_tf, y_scaled, training=False)
            
            # Calcolo Loss
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            target = tf.reduce_sum(log_prob) * LIKELIHOOD_SCALE

        # 3. Calcolo Gradiente rispetto a y_tf (Float32)
        grad = tape.gradient(target, y_tf)
        
        # Safety Check: Se il grafo è ancora rotto, stampa un warning (ma ora non dovrebbe succedere)
        if grad is None:
            tf.print("WARNING: Gradient is None! Check graph connection.")
            grad = tf.zeros_like(y_tf)
            
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
        outputs[0][0] = np.array(logp_val.numpy(), dtype=np.float64)
        
    def grad(self, inputs, output_gradients):
        y = inputs[0]
        return [output_gradients[0] * self.grad_op(y)]
    
# ==========================================
# 3. ESECUZIONE ADVI
# ==========================================

def print_uncertainty_report(trace, y_true):
    """
    Stampa una tabella con Media e Deviazione Standard per ogni classe.
    """
    posterior = trace.posterior['label_vector'].values.reshape(-1, NUM_CLASSES)
    
    mean_est = np.mean(posterior, axis=0)
    std_est = np.std(posterior, axis=0)
    
    print("\n" + "="*55)
    print(f"   REPORT INCERTEZZA ADVI (Scala Likelihood: {LIKELIHOOD_SCALE})")
    print("="*55)
    print(f"{'Classe':<8} | {'Vero':<10} | {'Pred (Media)':<12} | {'Std Dev':<10}")
    print("-" * 55)
    
    for i in range(NUM_CLASSES):
        print(f"{i+1:<8} | {y_true[i]:<10.4f} | {mean_est[i]:<12.4f} | {std_est[i]:<10.4f}")
    
    print("-" * 55)
    print("Nota: Std Dev alta (>0.05) indica incertezza del modello.")
    print("="*55 + "\n")

if __name__ == "__main__":
    print("--- 1. Caricamento Dati ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError("Mancano i file di normalizzazione (z_params o y_scale). Esegui training.")

    z_params = np.load(PARAMS_Z_PATH)
    z_mean = z_params['mean']
    z_std = z_params['std']
    y_scale_factor = np.load(PARAMS_Y_PATH)
    print(f"Parametri caricati: Z_mean {z_mean.shape}, Y_scale {y_scale_factor:.4f}")

    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    try:
        classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)
        print("Classificatore CVAE caricato per inizializzazione smart.")
    except:
        print("ERRORE: Classificatore non trovato.")
        exit()

    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, activation=ACTIVATION)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    realnvp.load_weights(WEIGHTS_PATH_MINMAX)
    
    # Selezione Campione
    IDX = np.random.randint(0, len(X))
    #IDX = 1225
    #IDX = 990

    print(f"\n--- Analisi ADVI (Variational Inference) su Campione {IDX} ---")
    
    X_sample = X[IDX:IDX+1]
    y_true = y[IDX]
    print(f"Label Vera: {y_true}")
    
    z_sample_raw, _, _ = encoder.predict(X_sample, verbose=0)
    
    # Normalizzazione Z
    z_sample_norm = (z_sample_raw - z_mean) / z_std
    
    # 1. Commenta la parte vecchia (CVAE):
    y_pred_init = classifier_cvae.predict(z_sample_raw, verbose=0)[0]
    y_pred_init = np.clip(y_pred_init, 0.01, 0.54)
    init_vals = {'label_vector': y_pred_init}
    print(f"Starting Point (CVAE): {y_pred_init}")

    # 2. Usa questa riga per partire dal CENTRO esatto (0.275):
    # Questo è un prior "neutro": assume che tutte le classi abbiano danno medio.
    # Se il modello converge al valore vero (es. 0 per le sane, alto per le danneggiate),
    # significa che il RealNVP funziona DAVVERO.
    #init_vals = {'label_vector': np.full((NUM_CLASSES,), 0.275)}
    #print(f"Starting Point (BLIND): {init_vals['label_vector']}")

    logp_op = RealNVPLLogpOp(realnvp, z_sample_norm, y_scale_factor)
    
    print("\n[Start] Esecuzione ADVI...")
    start_time = time.time()
    
    with pm.Model() as model:
        # Prior su scala REALE
        label_vector = pm.Uniform('label_vector', lower=0.0, upper=0.55, shape=(NUM_CLASSES,))
        
        pm.Potential("likelihood", logp_op(label_vector))
        
        # 1. Fit (Ottimizzazione Variazionale)
        approx = pm.fit(
            n=ITERATIONS, 
            method='fullrank_advi', 
            start=init_vals, 
            progressbar=True,
            obj_optimizer=pm.adam(learning_rate=0.01)
        )
        
        # 2. Sample
        trace = approx.sample(draws=10000)
        
    elapsed = time.time() - start_time
    print(f"[End] ADVI completato in {elapsed:.2f} secondi.")

    print_uncertainty_report(trace, y_true)

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    print("\n--- Generazione Grafici ---")
    
    posterior_flat = trace.posterior['label_vector'].values.reshape(-1, NUM_CLASSES)
    mean_est = np.mean(posterior_flat, axis=0)
    
    # 1. Istogrammi
    plt.figure(figsize=(15, 10))
    for i in range(NUM_CLASSES):
        plt.subplot(2, 4, i + 1)
        plt.hist(posterior_flat[:, i], bins=50, alpha=0.7, color='green', edgecolor='black', range=(0, 0.6), density=True)
        
        plt.axvline(y_true[i], color='red', linestyle='--', linewidth=3, label='True')
        plt.axvline(mean_est[i], color='blue', linestyle='-', linewidth=2, label='Mean')
        
        plt.title(f'Class {i + 1}\nTrue: {y_true[i]:.2f} | Est: {mean_est[i]:.2f}', fontsize=12)
        plt.xlabel('Damage Level')
        plt.xlim(0, 0.6)
        
        if i == 0: plt.ylabel('Frequency (Density)')
        if i == 6: plt.legend()
        
    plt.suptitle(f'ADVI Posterior Distribution - Sample {IDX}\nTime: {elapsed:.2f}s', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"advi_histograms_{IDX}.png"))
    plt.show()

    # 2. ELBO Plot
    plt.figure(figsize=(8, 5))
    plt.plot(approx.hist)
    plt.title("ADVI Loss (ELBO) Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, f"advi_elbo_{IDX}.png"))