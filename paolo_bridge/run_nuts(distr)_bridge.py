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
# 1. CONFIGURAZIONE NUTS BRIDGE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")

# Percorsi Modelli Bridge
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_bridge.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_bridge.keras") 
WEIGHTS_PATH_MINMAX = os.path.join(OUTPUT_DIR, "realnvp_bridge_weights.weights.h5")

# Parametri Normalizzazione Bridge (Creati in train_realnvp_bridge.py)
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_params_bridge.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor_bridge.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_nuts_bridge") 
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Import loader bridge
from data_loader_bridge import load_and_process_data, DATA_DIR_TEST, CHANNELS, SEQ_LEN, N_CLASS

# Configurazione Parametri (Identici al training bridge)
LATENT_DIM = 4
NUM_CLASSES = N_CLASS # 6 Classi
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 64
ACTIVATION = "relu"
CORES = 1

# Tuning NUTS
DRAWS = 1000       
TUNE = 500        
CHAINS = 2         
LIKELIHOOD_SCALE = 4.0 
TARGET_ACCEPT = 0.8
STEP_SCALE = 0.01

# ==========================================
# 2. DEFINIZIONE MODELLO (BRIDGE TF-PYMC)
# ==========================================

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CouplingMasked(layers.Layer):
    def __init__(self, latent_dim, num_classes, reg=0.001):
        super().__init__()
        
        # 1. LABEL EMBEDDING (Deve matchare train_realnvp_bridge)
        self.label_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu')
        ])

        # 2. RAMO T (Traslazione) - Struttura identica al training
        self.t_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        # Nota: In inferenza il Dropout non serve dichiararlo se non lo usiamo, 
        # ma per caricare i pesi senza errori è meglio che i layer Dense abbiano gli stessi nomi/ordine
        self.t_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.t_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_out     = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg))

        # 3. RAMO S (Scalatura)
        self.s_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
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
            term_to_transform = z * reversed_mask
            transformed_term = term_to_transform * tf.exp(s) + t
            z = (z * mask) + (transformed_term * reversed_mask)
            log_det_inv += tf.reduce_sum(s * reversed_mask, axis=-1)
        return z, log_det_inv

    @tf.function
    def value_and_gradient(self, x, y, y_scale_factor_tf):
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        
        if len(x.shape) == 1: x = tf.expand_dims(x, 0)
        if len(y.shape) == 1: y = tf.expand_dims(y, 0)

        with tf.GradientTape() as tape:
            tape.watch(y)
            
            # --- SCALING Y ---
            # Applica lo scale factor caricato da train_realnvp_bridge
            y_scaled = y * y_scale_factor_tf
            
            z_pred, log_det_inv = self(x, y_scaled)
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            target = tf.reduce_sum(log_prob) * LIKELIHOOD_SCALE

        grad = tape.gradient(target, y)
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

def analyze_divergences(trace):
    print("\n" + "="*60)
    print("   ANALISI DELLE DIVERGENZE (Betancourt & Girolami)")
    print("="*60)
    
    diverging = trace.sample_stats["diverging"].values.flatten()
    n_divergences = np.sum(diverging)
    print(f"Totale Divergenze Rilevate: {n_divergences}")
    
    if n_divergences > 0:
        samples_flat = trace.posterior["label_vector"].values.reshape(-1, NUM_CLASSES)
        divergent_samples = samples_flat[diverging]
        divergent_classes = np.argmax(divergent_samples, axis=1)
        counts = np.bincount(divergent_classes, minlength=NUM_CLASSES)
        
        print("\nLe divergenze si concentrano su queste classi:")
        for i in range(NUM_CLASSES):
            if counts[i] > 0:
                print(f"Classe {i+1}: {counts[i]} divergenze")
        winner_div = np.argmax(counts)
        return winner_div
    else:
        print("Nessuna divergenza rilevata.")
        return None

# ==========================================
# 3. ESECUZIONE NUTS
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati e Parametri Bridge ---")
    # Caricamento dati Test Bridge
    X, y = load_and_process_data(DATA_DIR_TEST, is_training=False)
    
    # Check File Esistenza
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError(f"Mancano i file di normalizzazione in {OUTPUT_DIR}. Esegui train_realnvp_bridge.py.")

    # Carica parametri Normalizzazione e Scala
    z_params = np.load(PARAMS_Z_PATH)
    z_mean = z_params['mean']
    z_std = z_params['std']
    y_scale_factor = np.load(PARAMS_Y_PATH)
    print(f"Parametri caricati: Z_mean {z_mean.shape}, Y_scale {y_scale_factor:.4f}")

    # Carica Modelli Keras
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)

    # Inizializza RealNVP Bridge
    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, activation=ACTIVATION)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES))) # Build
    realnvp.load_weights(WEIGHTS_PATH_MINMAX)
    print("Pesi RealNVP Bridge caricati con successo.")

    # Selezione Campione (Random o Fisso)
    IDX = np.random.randint(0, len(X))
    IDX = 819
    print(f"\n--- Analisi MCMC NUTS su Campione Bridge {IDX} ---")
    
    X_sample = X[IDX:IDX+1]
    y_true = y[IDX]
    
    # 1. Encoding RAW
    z_sample_raw, _, _ = encoder.predict(X_sample, verbose=0)
    
    # 2. Normalizzazione Z (Con parametri Bridge)
    z_sample_norm = (z_sample_raw - z_mean) / z_std
    
    # 3. Start Point CVAE (Predizione Iniziale)
    y_pred_init = classifier_cvae.predict(z_sample_raw, verbose=0)[0]
    # Clip per restare nel dominio fisico (assumendo danni simili al caso precedente)
    y_pred_init = np.clip(y_pred_init, 0.001, 0.54) 
    
    print(f"Punto di partenza (CVAE): {y_pred_init}")
    print(f"Punto true : {y_true}")

    start_vals = []
    for _ in range(CHAINS):
        start_vals.append({'label_vector': y_pred_init})

    # Op Theano per PyMC
    logp_op = RealNVPLLogpOp(realnvp, z_sample_norm, y_scale_factor)
    
    print("\n[Start] Campionamento NUTS...")
    start_time = time.time()
    
    y_init_safe = np.clip(y_pred_init / 0.55, 0.01, 0.99)
    y_init_logit = np.log(y_init_safe / (1 - y_init_safe))

    with pm.Model() as model:
        # 1. Spazio Illimitato: Distribuzione Logistica
        # Usando mu=0, s=1 otteniamo una Uniforme perfetta dopo la trasformazione.
        # Se aumenti 's' (es. s=2), spingi verso gli estremi (0 e 0.55).
        #y_logit = pm.Logistic('y_logit', mu=y_init_logit, s=0.5,shape=(NUM_CLASSES,))

        # 2. Trasformazione "Soft" nel range fisico [0, 0.55]
        # Questo crea un "muro di gomma": avvicinandosi a 0.55 la pendenza cambia,
        # permettendo a NUTS di rallentare invece di schiantarsi.
        #label_vector = pm.Deterministic('label_vector', 0.55 * pm.math.sigmoid(y_logit))
        
        
        y_logit = pm.Logistic('y_logit', mu=y_init_logit, s=1, shape=(NUM_CLASSES,))

        # ---------------------------------------------------------
        # B. TRASFORMAZIONE
        # ---------------------------------------------------------
        # La Sigmoide porta in [0, 1], poi moltiplichiamo per 0.55
        label_vector = pm.Deterministic('label_vector', 0.55 * pm.math.sigmoid(y_logit))
        #label_vector = pm.Uniform('label_vector', lower=0.0, upper=0.55, shape=(NUM_CLASSES,))
        pm.Potential("likelihood", logp_op(label_vector))

        trace = pm.sample(
                    draws=DRAWS, 
                    tune=TUNE, 
                    chains=CHAINS, 
                    cores=CORES, 
                    #init='adapt_diag', 
                    # Importante: passiamo lo start point nello spazio LOGIT
                    #initvals=[{'y_logit': y_init_logit} for _ in range(CHAINS)], y_pred_init
                    initvals=[{'label_vector': y_pred_init} for _ in range(CHAINS)],
                    step=pm.NUTS(
                        target_accept=TARGET_ACCEPT, # 0.99 forza passi piccolissimi
                        step_scale=STEP_SCALE,       # Suggerimento iniziale: "parti piano"
                        max_treedepth=12             # Aumenta se i passi sono troppo piccoli e non esplora
                    ),
                    progressbar=True,
                    discard_tuned_samples=True
                )
        
    elapsed = time.time() - start_time
    print(f"[End] MCMC NUTS completato in {elapsed:.2f} secondi.")

    # ==========================================
    # 4. PLOTTING E DIAGNOSTICA
    # ==========================================

    hidden_winner = analyze_divergences(trace)

    print("\n--- Generazione Grafici e Diagnostica ---")
    
    print(az.summary(trace, round_to=2))
    
    posterior_flat = trace.posterior['label_vector'].values.reshape(-1, NUM_CLASSES)
    mean_est = np.mean(posterior_flat, axis=0)
    
    # Plot per N_CLASS (6)
    plt.figure(figsize=(15, 8))
    for i in range(NUM_CLASSES):
        plt.subplot(2, 3, i + 1) # Adattato per 6 classi (2 righe, 3 colonne)
        
        # Istogramma
        plt.hist(posterior_flat[:, i], bins=50, alpha=0.7, color='firebrick', edgecolor='black', density=True)
        plt.xlim(0.0, 0.6) 
        
        # Linee
        plt.axvline(y_true[i], color='blue', linestyle='--', linewidth=3, label='True')
        plt.axvline(mean_est[i], color='gold', linestyle='-', linewidth=2, label='Post Mean')
        
        # HDI
        hdi = az.hdi(trace.posterior['label_vector'].sel(label_vector_dim_0=i), hdi_prob=0.94)
        hdi_val = hdi['label_vector'].values
        plt.hlines(y=0.5, xmin=hdi_val[0], xmax=hdi_val[1], color='black', linewidth=4, label='94% HDI')
        
        plt.title(f'Class {i + 1}\nTrue: {y_true[i]:.2f} | Est: {mean_est[i]:.2f}', fontsize=10)
        if i == 0: plt.ylabel('Density')
        if i == 2: plt.legend(loc='upper right', fontsize='x-small')
        
    plt.suptitle(f'NUTS Bridge MCMC - Sample {IDX}\nDraws: {DRAWS*CHAINS} | Time: {elapsed:.2f}s', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"nuts_bridge_histograms_{IDX}.png"))
    plt.show()

    az.plot_trace(trace)
    plt.savefig(os.path.join(RESULTS_DIR, f"nuts_bridge_traceplot_{IDX}.png"))

    print("\n--- SINTESI ESPLORAZIONE NUTS ---")
    print(az.summary(trace, var_names=['label_vector'], round_to=4))

    print(f"\nRisultati salvati in: {RESULTS_DIR}")