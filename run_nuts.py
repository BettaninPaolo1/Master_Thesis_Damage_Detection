import os
import time
import numpy as np
import pandas as pd
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
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, r2_score
import seaborn as sns

# ==========================================
# 1. CONFIGURAZIONE NUTS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_cvae.keras")
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "realnvp_weights.weights.h5")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_nuts")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

LATENT_DIM = 4
NUM_CLASSES = 7
NUM_COUPLING_LAYERS = 6
HIDDEN_DIM = 64

# --- TUNING NUTS ---
# NUTS è molto più efficiente di Metropolis, servono meno campioni
NUM_TEST_SAMPLES = 50   
DRAWS = 500             # 500 campioni efficaci NUTS valgono come 10k Metropolis
TUNE = 500              # Warmup per adattare il passo
CHAINS = 2              # 2 catene per verificare la convergenza (R-hat)
CORES = 1               # 1 Core per evitare conflitti TF/PyMC
LIKELIHOOD_SCALE = 10.0 

# ==========================================
# 2. DEFINIZIONE MODELLI E PONTE GRADIENTI
# ==========================================

class CouplingMasked(layers.Layer):
    def __init__(self, latent_dim, num_classes, hidden_dim=64, reg=0.01):
        super().__init__()
        self.t_net = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(hidden_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg)) 
        ])
        self.s_net = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(hidden_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(latent_dim, activation="tanh", kernel_regularizer=regularizers.l2(reg)) 
        ])

    def call(self, x_masked, y):
        combined = layers.concatenate([x_masked, y], axis=-1)
        return self.s_net(combined), self.t_net(combined)

class RealNVP_Full(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim):
        super().__init__()
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        self.masks = tf.constant([[1, 1, 0, 0] if i % 2 == 0 else [0, 0, 1, 1] for i in range(num_coupling_layers)], dtype=tf.float32)
        self.layers_list = [CouplingMasked(latent_dim, num_classes, hidden_dim) for _ in range(num_coupling_layers)]

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
    def value_and_gradient(self, x, y):
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        
        # 1. Assicuriamo che X abbia la dimensione batch (1, 4)
        if len(x.shape) == 1:
            x = tf.expand_dims(x, 0)
        
        with tf.GradientTape() as tape:
            tape.watch(y)
            
            # 2. FIX CRITICO: Aggiungiamo dimensione batch anche a Y -> (1, 7)
            # PyMC passa (7,), ma Concatenate vuole (1, 7)
            y_in = y
            if len(y.shape) == 1:
                y_in = tf.expand_dims(y, 0)
            
            # Passiamo y_in (con batch) al modello
            z_pred, log_det_inv = self(x, y_in, training=False)
            
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            target = tf.reduce_sum(log_prob) * LIKELIHOOD_SCALE
        
        # Calcoliamo il gradiente rispetto all'originale 'y' (senza batch)
        grad = tape.gradient(target, y)
        
        # Gestione sicurezza
        if grad is None:
            grad = tf.zeros_like(y)
        grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
        
        return target, grad

# --- OPS PYTENSOR (IL PONTE) ---
class RealNVPGradientOp(Op):
    def __init__(self, model, x_fixed):
        self.model = model; self.x_fixed = x_fixed 
    def make_node(self, y): return Apply(self, [pt.as_tensor_variable(y)], [pt.as_tensor_variable(y).type()])
    def perform(self, node, inputs, outputs):
        y_val = inputs[0]
        _, grad_val = self.model.value_and_gradient(self.x_fixed, y_val)
        outputs[0][0] = grad_val.numpy().astype(np.float64).flatten()

class RealNVPLLogpOp(Op):
    def __init__(self, model, x_fixed):
        self.model = model; self.x_fixed = x_fixed; self.grad_op = RealNVPGradientOp(model, x_fixed)
    def make_node(self, y): return Apply(self, [pt.as_tensor_variable(y)], [pt.dscalar()]) 
    def perform(self, node, inputs, outputs):
        v, _ = self.model.value_and_gradient(self.x_fixed, inputs[0])
        outputs[0][0] = np.array(v.numpy(), dtype=np.float64)
    def grad(self, i, o): 
        # Qui diciamo a PyMC come calcolare il gradiente della Likelihood
        return [o[0] * self.grad_op(i[0])]

# ==========================================
# 3. VALUTAZIONE E PLOTTING
# ==========================================
def evaluate_metrics(results_df):
    print("\n" + "="*50)
    print("   REPORT DI VALUTAZIONE NUTS (Gradient MCMC)")
    print("="*50)

    def parse_col(col_name):
        if isinstance(results_df[col_name].iloc[0], str):
            return np.array([np.fromstring(x.strip('[]'), sep=' ') for x in results_df[col_name]])
        return np.stack(results_df[col_name].values)

    y_true_all = parse_col('True_Label')
    y_pred_all = parse_col('Pred_Mean')
    y_vae_all  = parse_col('VAE_Init')

    true_classes = np.argmax(y_true_all, axis=1) + 1
    pred_classes = np.argmax(y_pred_all, axis=1) + 1
    true_levels = np.max(y_true_all, axis=1)
    pred_levels = y_pred_all[np.arange(len(y_pred_all)), np.argmax(y_pred_all, axis=1)]
    vae_levels = y_vae_all[np.arange(len(y_vae_all)), np.argmax(y_vae_all, axis=1)]

    acc = accuracy_score(true_classes, pred_classes)
    mae_level = mean_absolute_error(true_levels, pred_levels)
    r2 = r2_score(true_levels, pred_levels)
    
    print(f"Campioni Totali: {len(results_df)}")
    print(f"ACCURACY Classificazione: {acc:.4f} ({acc*100:.2f}%)")
    print(f"MAE Livello Danno:        {mae_level:.4f}")
    print(f"R2 Score:                 {r2:.4f}")

    # PLOT 1: Confusion Matrix
    cm = confusion_matrix(true_classes, pred_classes, labels=range(1, 8))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=range(1, 8), yticklabels=range(1, 8))
    plt.title(f'Confusion Matrix - NUTS\nAccuracy: {acc:.2f} | MAE: {mae_level:.3f}')
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.show()

    # PLOT 2: Regression Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(true_levels, pred_levels, alpha=0.7, color='darkorange', edgecolors='k', s=80, label='NUTS Prediction')
    plt.plot([0, 0.6], [0, 0.6], 'k--', linewidth=2, label='Perfect Fit')
    plt.title(f'Regression Analysis: True vs NUTS\nR2 Score: {r2:.3f}')
    plt.xlabel('True Damage Level')
    plt.ylabel('Predicted Damage Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(RESULTS_DIR, "regression_plot.png"))
    plt.show()

    # PLOT 3: VAE vs NUTS Shift
    plt.figure(figsize=(10, 8))
    plt.scatter(true_levels, vae_levels, marker='x', color='gray', s=60, label='VAE Initial Guess', alpha=0.6)
    plt.scatter(true_levels, pred_levels, marker='o', color='red', s=60, label='NUTS Refined', edgecolors='k')
    
    for i in range(len(true_levels)):
        if abs(pred_levels[i] - vae_levels[i]) > 0.005:
            plt.plot([true_levels[i], true_levels[i]], [vae_levels[i], pred_levels[i]], color='gray', alpha=0.3)

    plt.plot([0, 0.6], [0, 0.6], 'k--', linewidth=2)
    plt.title('Impact of NUTS Inference (Shift from VAE)')
    plt.xlabel('True Damage Level')
    plt.ylabel('Estimated Damage Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(RESULTS_DIR, "vae_vs_nuts_shift.png"))
    plt.show()

# ==========================================
# 4. MAIN LOOP
# ==========================================
if __name__ == "__main__":
    print("--- 1. Caricamento Dati (TEST SET) ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
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
        print("Classificatore CVAE caricato.")
    except:
        print("WARN: Classificatore non trovato.")
        exit()

    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    realnvp.load_weights(WEIGHTS_PATH)
    
    indices = np.random.choice(len(X), NUM_TEST_SAMPLES, replace=False)
    results_list = []
    
    print(f"\n--- 3. Inizio Loop NUTS su {NUM_TEST_SAMPLES} campioni ---")
    
    for i, idx in enumerate(indices):
        print(f"[{i+1}/{NUM_TEST_SAMPLES}] NUTS Index {idx}...", end="", flush=True)
        start_t = time.time()
        
        try:
            X_sample = X[idx:idx+1]
            y_true = y[idx]
            z_mean, _, _ = encoder.predict(X_sample, verbose=0)
            
            # Smart Init
            y_pred_init = np.clip(classifier_cvae.predict(z_mean, verbose=0)[0], 0.01, 0.54)
            init_vals = {'label_vector': y_pred_init}
            
            logp_op = RealNVPLLogpOp(realnvp, z_mean)
            
            with pm.Model() as model:
                # Prior Uniforme (come negli altri script)
                label_vector = pm.Uniform('label_vector', lower=0.0, upper=0.55, shape=(NUM_CLASSES,))
                pm.Potential("likelihood", logp_op(label_vector))
                
                # NUTS: Usa il gradiente automaticamente grazie all'Op definita sopra
                trace = pm.sample(
                    draws=DRAWS, 
                    tune=TUNE, 
                    chains=CHAINS, 
                    cores=CORES,
                    initvals=init_vals,
                    step=pm.NUTS(), 
                    progressbar=False,
                    discard_tuned_samples=True
                )
            
            post = trace.posterior['label_vector'].values.reshape(-1, NUM_CLASSES)
            mean_est = np.mean(post, axis=0)
            std_est = np.std(post, axis=0)
            
            elapsed = time.time() - start_t
            
            results_list.append({
                "Index": idx, 
                "True_Label": y_true, 
                "Pred_Mean": mean_est,
                "Pred_Std": std_est,
                "VAE_Init": y_pred_init,
                "Time": elapsed
            })
            
            # R-Hat check (opzionale per debug)
            try:
                rhat = az.rhat(trace)['label_vector'].max().values
                print(f" Done ({elapsed:.2f}s) | R-hat: {rhat:.3f}")
            except:
                print(f" Done ({elapsed:.2f}s).")

            if (i+1) % 5 == 0:
                pd.DataFrame(results_list).to_csv(os.path.join(RESULTS_DIR, "results_nuts.csv"), index=False)
                
        except Exception as e:
            print(f" ERR: {e}")
            
    final_df = pd.DataFrame(results_list)
    final_df.to_csv(os.path.join(RESULTS_DIR, "results_nuts.csv"), index=False)
    evaluate_metrics(final_df)