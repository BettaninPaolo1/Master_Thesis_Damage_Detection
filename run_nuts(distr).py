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
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "realnvp_weights.weights.h5")
WEIGHTS_PATH_MINMAX= os.path.join(OUTPUT_DIR, "realnvp_minmax14layers_weights.weights.h5")

# PERCORSI PARAMETRI DI NORMALIZZAZIONE (NUOVI)
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_normalization_params.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_nuts(distr)") 

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR, TEST_DIR

# Configurazione Parametri NUTS
LATENT_DIM = 4
NUM_CLASSES = 7
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 64
ACTIVATION = "relu" # <--- Importante: deve matchare il training!

# Tuning NUTS
  
STEP_SCALE = 0.05
DRAWS = 500       
TUNE = 500        
CHAINS = 2         
LIKELIHOOD_SCALE = 8.0 
TARGET_ACCEPT = 0.8

# ==========================================
# 2. DEFINIZIONE MODELLO (BRIDGE TF-PYMC)
# ==========================================

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
            
            # --- SCALING Y (CRUCIALE) ---
            # NUTS propone valori "reali" (es. 0.3).
            # Il modello vuole valori scalati (es. 0.3 * factor).
            y_scaled = y * y_scale_factor_tf
            
            z_pred, log_det_inv = self(x, y_scaled, training=False)
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
        # Passiamo anche il fattore di scala!
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
        # Passiamo anche il fattore di scala!
        logp_val, _ = self.model.value_and_gradient(self.x_fixed, y_val, self.y_scale_factor)
        outputs[0][0] = np.array(logp_val.numpy(), dtype=np.float64)
        
    def grad(self, inputs, output_gradients):
        y = inputs[0]
        return [output_gradients[0] * self.grad_op(y)]

def analyze_divergences(trace):
    print("\n" + "="*60)
    print("   ANALISI DELLE DIVERGENZE (Betancourt & Girolami)")
    print("="*60)
    
    # 1. Estrai le posizioni dove sono avvenute le divergenze
    # PyMC salva le divergenze in trace.sample_stats.diverging
    diverging = trace.sample_stats["diverging"].values.flatten()
    n_divergences = np.sum(diverging)
    
    print(f"Totale Divergenze Rilevate: {n_divergences}")
    
    if n_divergences > 0:
        # Estrai i campioni corrispondenti alle divergenze
        # Nota: le divergenze indicano il punto di partenza del salto fallito
        samples_flat = trace.posterior["label_vector"].values.reshape(-1, 7)
        divergent_samples = samples_flat[diverging]
        
        # 2. Dove puntavano queste divergenze?
        # Calcoliamo la classe media verso cui si stava dirigendo il sistema
        avg_divergent_pos = np.mean(divergent_samples, axis=0)
        
        # O meglio: Quale classe era attiva nei punti che hanno causato il crash?
        # Spesso il crash avviene proprio sul bordo del "collo dell'imbuto" (Classe 4)
        divergent_classes = np.argmax(divergent_samples, axis=1)
        counts = np.bincount(divergent_classes, minlength=7)
        
        print("\nLe divergenze si concentrano su queste classi:")
        for i in range(7):
            if counts[i] > 0:
                print(f"Classe {i+1}: {counts[i]} divergenze")
                
        winner_div = np.argmax(counts)
        print(f"\nINDIZIO NASCOSTO: Le divergenze suggeriscono problemi di geometria sulla Classe {winner_div+1}.")
        
        return winner_div
    else:
        print("Nessuna divergenza rilevata. L'esplorazione geometrica è stata stabile.")
        return None

# ==========================================
# 3. ESECUZIONE NUTS
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati e Parametri ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    # Check File Esistenza
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError("Mancano i file di normalizzazione (z_params o y_scale). Esegui training.")

    # Carica parametri
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
    classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)

    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, activation=ACTIVATION)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    realnvp.load_weights(WEIGHTS_PATH_MINMAX)
    
    # Selezione Campione
    IDX = np.random.randint(0, len(X))
    #IDX = 2237 # Fisso per debug
    IDX = 2752
    IDX = 2955
    
    
    print(f"\n--- Analisi MCMC NUTS (Corrected) su Campione {IDX} ---")
    
    X_sample = X[IDX:IDX+1]
    y_true = y[IDX]
    
    # 1. Encoding RAW
    z_sample_raw, _, _ = encoder.predict(X_sample, verbose=0)
    
    # 2. Normalizzazione Z (IMPORTANTE!)
    z_sample_norm = (z_sample_raw - z_mean) / z_std
    
    # 3. Start Point CVAE
    # Il classificatore lavora su Z raw solitamente (verifica il training del classificatore)
    # Se il classificatore è stato trainato su Z raw, usiamo raw.
    y_pred_init = classifier_cvae.predict(z_sample_raw, verbose=0)[0]
    y_pred_init = np.clip(y_pred_init, 0.001, 0.549)
    
    print(f"Punto di partenza (CVAE): {y_pred_init}")

    start_vals = []
    for _ in range(CHAINS):
        start_vals.append({'label_vector': y_pred_init})

    # Passiamo Z NORMALIZZATO e il fattore di scala Y all'Op Theano/PyTensor
    logp_op = RealNVPLLogpOp(realnvp, z_sample_norm, y_scale_factor)
    
    print("\n[Start] Campionamento NUTS...")
    start_time = time.time()

    y_init_safe = np.clip(y_pred_init / 0.55, 0.01, 0.99)
    y_init_logit = np.log(y_init_safe / (1 - y_init_safe))

    with pm.Model() as model:
        y_logit = pm.Logistic('y_logit', mu=0, s=1, shape=(NUM_CLASSES,)) # metti 0 al posto di y_init_logit per aver un uniforme
        label_vector = pm.Deterministic('label_vector', 0.55 * pm.math.sigmoid(y_logit))
        # Prior su scala REALE (0 - 0.55)
        #label_vector = pm.Uniform('label_vector', lower=0.0, upper=0.55, shape=(NUM_CLASSES,))
        
        # Likelihood custom
        pm.Potential("likelihood", logp_op(label_vector))

        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,       
            chains=CHAINS,
            cores=1,         
            start=start_vals,  
            init='adapt_diag',
            step=pm.NUTS(
            target_accept=TARGET_ACCEPT, # piu alto è più forza passi piccoli
            step_scale=STEP_SCALE,       # Suggerimento iniziale: "parti piano"
            max_treedepth=12),             # Aumenta se i passi sono troppo piccoli e non esplora),
            discard_tuned_samples=False,
            progressbar=True
        )
        
    elapsed = time.time() - start_time
    print(f"[End] MCMC NUTS completato in {elapsed:.2f} secondi.")

    # ==========================================
    # 4. PLOTTING E DIAGNOSTICA
    # ==========================================

    hidden_winner = analyze_divergences(trace)

    print("\n--- Generazione Grafici e Diagnostica ---")
    
    print(az.summary(trace, round_to=2))
    
    posterior_flat = trace.posterior['label_vector'].values
    posterior_flat = posterior_flat.reshape(-1, NUM_CLASSES)
    mean_est = np.mean(posterior_flat, axis=0)
    
    plt.figure(figsize=(15, 10))
    for i in range(NUM_CLASSES):
        plt.subplot(2, 4, i + 1)
        
        # Istogramma
        plt.hist(posterior_flat[:, i], bins=50, alpha=0.7, color='royalblue', edgecolor='black', density=True)
        
        # Fissiamo i limiti per rendere i grafici confrontabili
        plt.xlim(0.0, 0.6) 
        
        # Linee di riferimento
        plt.axvline(y_true[i], color='red', linestyle='--', linewidth=3, label='True')
        plt.axvline(mean_est[i], color='orange', linestyle='-', linewidth=2, label='Post Mean')
        
        # HDI
        hdi = az.hdi(trace.posterior['label_vector'].sel(label_vector_dim_0=i), hdi_prob=0.94)
        hdi_val = hdi['label_vector'].values
        plt.hlines(y=0.5, xmin=hdi_val[0], xmax=hdi_val[1], color='black', linewidth=4, label='94% HDI')
        
        plt.title(f'Class {i + 1}\nTrue: {y_true[i]:.2f} | Est: {mean_est[i]:.2f}', fontsize=12)
        plt.xlabel('Damage Level')
        if i == 0: plt.ylabel('Density')
        if i == 6: plt.legend(loc='upper right', fontsize='small')
        
    plt.suptitle(f'NUTS MCMC Posterior - Sample {IDX}\nDraws: {DRAWS*CHAINS} | Time: {elapsed:.2f}s', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"nuts_histograms_{IDX}.png"))
    plt.show()

    az.plot_trace(trace)
    plt.savefig(os.path.join(RESULTS_DIR, f"nuts_traceplot_{IDX}.png"))

    print(f"\nRisultati salvati in: {RESULTS_DIR}")