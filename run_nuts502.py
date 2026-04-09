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
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CONFIGURAZIONE NUTS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_cvae.keras")

WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "realnvp_minmax14layers_weights.weights.h5")
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_normalization_params.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_nuts_stats4") # Cartella dedicata

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

LATENT_DIM = 4
NUM_CLASSES = 7
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 64
ACTIVATION = "relu"

# --- TUNING NUTS ---
NUM_TEST_SAMPLES = 500
DRAWS = 500            
TUNE = 500              
CHAINS = 2           
CORES = 1               
LIKELIHOOD_SCALE = 8.0 
TARGET_ACCEPT = 0.8
STEP_SCALE = 0.01

# ==========================================
# 2. DEFINIZIONE MODELLI (CORRETTA CON DROPOUT)
# ==========================================
try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

# Parametro Globale (deve essere lo stesso del training)
DROPOUT_RATE = 0.1 

@register_serializable()
class CouplingMasked(layers.Layer):
    def __init__(self, latent_dim, num_classes, hidden_dim=64, reg=0.01, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.reg = reg
        self.hidden_dim = hidden_dim
        
        # 1. LABEL EMBEDDING
        self.label_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu')
        ], name="label_embedding")

        # 2. RAMO T (Traslazione)
        self.t_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_dropout = layers.Dropout(DROPOUT_RATE) # <--- MANCAVA QUESTO
        self.t_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.t_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_out     = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg))

        # 3. RAMO S (Scalatura)
        self.s_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_dropout = layers.Dropout(DROPOUT_RATE) # <--- MANCAVA QUESTO
        self.s_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.s_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_out     = layers.Dense(latent_dim, activation="tanh", kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y, training=False):
        # A. Embedding
        label_emb = self.label_net(y)

        # B. Ramo T
        t_z = self.t_z_dense(x_masked)
        t_z = self.t_dropout(t_z, training=training) # Importante passare training
        t_l = self.t_l_dense(label_emb)
        
        t_concat = layers.Concatenate()([t_z, t_l])
        t = self.t_out(self.t_joint(t_concat))

        # C. Ramo S
        s_z = self.s_z_dense(x_masked)
        s_z = self.s_dropout(s_z, training=training) # Importante passare training
        s_l = self.s_l_dense(label_emb)
        
        s_concat = layers.Concatenate()([s_z, s_l])
        s = self.s_out(self.s_joint(s_concat))

        return s, t
    
    def get_config(self):
        return super().get_config() | {
            "latent_dim": self.latent_dim, 
            "num_classes": self.num_classes, 
            "reg": self.reg,
            "hidden_dim": self.hidden_dim
        }

@register_serializable()
class RealNVP_Full(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, activation=ACTIVATION, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim # Importante per get_config
        
        self.distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim))
        
        # Maschere alternate
        masks_list = [[1, 1, 0, 0] if i % 2 == 0 else [0, 0, 1, 1] for i in range(num_coupling_layers)]
        self.masks = tf.constant(masks_list, dtype=tf.float32)
        
        # Istanza layers corretti
        self.layers_list = [
            CouplingMasked(latent_dim, num_classes, hidden_dim=hidden_dim) 
            for _ in range(num_coupling_layers)
        ]

    def call(self, x, y):
        log_det_inv = 0
        z = x
        for i in range(self.num_coupling_layers):
            mask = self.masks[i]
            reversed_mask = 1 - mask
            z_masked = z * mask
            
            # Passiamo training=False per disattivare dropout in inferenza
            s, t = self.layers_list[i](z_masked, y, training=False)
            
            term_to_transform = z * reversed_mask
            transformed_term = term_to_transform * tf.exp(s) + t
            z = (z * mask) + (transformed_term * reversed_mask)
            log_det_inv += tf.reduce_sum(s * reversed_mask, axis=-1)
        return z, log_det_inv

    @tf.function
    def value_and_gradient(self, x, y, y_scale_factor_tf):
        # Casting e Reshape (Safe)
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        if len(x.shape) == 1: x = tf.expand_dims(x, 0)
        if len(y.shape) == 1: y = tf.expand_dims(y, 0)

        with tf.GradientTape() as tape:
            tape.watch(y)
            
            # SCALING: Fondamentale per far lavorare la rete nel suo range
            y_scaled = y * y_scale_factor_tf
            
            z_pred, log_det_inv = self(x, y_scaled, training=False)
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            target = tf.reduce_sum(log_prob) * LIKELIHOOD_SCALE
        
        grad = tape.gradient(target, y)
        
        # SALVAGENTE NAN: Fondamentale per NUTS
        if grad is None: 
            grad = tf.zeros_like(y)
        grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
        
        return target, grad
    
    def get_config(self):
        return super().get_config() | {
            "num_coupling_layers": self.num_coupling_layers, 
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim
        }
# ==========================================
# 3. OPS PYTENSOR
# ==========================================
class RealNVPGradientOp(Op):
    def __init__(self, model, x_fixed, y_scale_factor):
        self.model, self.x_fixed = model, x_fixed 
        self.y_scale_factor = tf.constant(y_scale_factor, dtype=tf.float32)
    def make_node(self, y): return Apply(self, [pt.as_tensor_variable(y)], [pt.as_tensor_variable(y).type()])
    def perform(self, node, inputs, outputs):
        _, grad_val = self.model.value_and_gradient(self.x_fixed, inputs[0], self.y_scale_factor)
        outputs[0][0] = grad_val.numpy().astype(np.float64).flatten()

class RealNVPLLogpOp(Op):
    def __init__(self, model, x_fixed, y_scale_factor):
        self.model, self.x_fixed = model, x_fixed
        self.y_scale_factor = tf.constant(y_scale_factor, dtype=tf.float32)
        self.grad_op = RealNVPGradientOp(model, x_fixed, y_scale_factor)
    def make_node(self, y): return Apply(self, [pt.as_tensor_variable(y)], [pt.dscalar()]) 
    def perform(self, node, inputs, outputs):
        v, _ = self.model.value_and_gradient(self.x_fixed, inputs[0], self.y_scale_factor)
        outputs[0][0] = np.array(v.numpy(), dtype=np.float64)
    def grad(self, i, o): return [o[0] * self.grad_op(i[0])]

# ==========================================
# 4. HELPER STATS
# ==========================================
def calculate_mode_vector(particles, lower=0.0, upper=0.55):
    modes, x_grid = [], np.linspace(lower, upper, 100)
    for i in range(particles.shape[1]):
        try: modes.append(x_grid[np.argmax(gaussian_kde(particles[:, i])(x_grid))])
        except: modes.append(np.mean(particles[:, i]))
    return np.array(modes)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100 * np.mean(np.abs(y_true - y_pred) / np.where(denom==0, 1.0, denom))

# ==========================================
# 5. MAIN LOOP
# ==========================================
if __name__ == "__main__":

    print("--- 1. Caricamento Dati e Parametri ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    z_params = np.load(PARAMS_Z_PATH)
    z_mean_arr, z_std_arr = z_params['mean'], z_params['std']
    y_scale_factor = np.load(PARAMS_Y_PATH)

    try: keras.utils.get_custom_objects().update({"CouplingMasked": CouplingMasked, "RealNVP_Full": RealNVP_Full})
    except: pass

    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, i): return i[0] + tf.exp(0.5 * i[1]) * tf.random.normal(tf.shape(i[0]))

    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)

    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    realnvp.load_weights(WEIGHTS_PATH)
    
    #np.random.seed(84)
    np.random.seed(67)
    indices = np.random.choice(len(X), min(NUM_TEST_SAMPLES, len(X)), replace=False)
    
    results_list = []
    csv_file = os.path.join(RESULTS_DIR, "results_nuts_batch.csv")
    
    print(f"\n--- 3. Inizio Loop NUTS su {NUM_TEST_SAMPLES} campioni ---")
    
    for i, idx in enumerate(indices):
        print(f"[{i+1}/{NUM_TEST_SAMPLES}] NUTS Index {idx}...", end="", flush=True)
        start_t = time.time()
        try:
            X_sample = X[idx:idx+1]
            y_true = y[idx]
            
            # 1. Encoding
            z_mean_raw, _, _ = encoder.predict(X_sample, verbose=0)
            
            # 2. Normalizzazione Z
            z_mean_norm = (z_mean_raw - z_mean_arr) / z_std_arr
            
            # 3. Inizializzazione CVAE
            # Correzione: Clip a 0.001 come nel file singolo (era 0.01)
            y_pred_init = np.clip(classifier_cvae.predict(z_mean_raw, verbose=0)[0], 0.001, 0.549)
            
            # Correzione: Creazione esplicita della lista di start values per ogni catena
            start_vals = [{'label_vector': y_pred_init} for _ in range(CHAINS)]

            logp_op = RealNVPLLogpOp(realnvp, z_mean_norm, y_scale_factor)

            y_init_safe = np.clip(y_pred_init / 0.55, 0.01, 0.99)
            y_init_logit = np.log(y_init_safe / (1 - y_init_safe))

            # 3. PyMC NUTS
            logp_op = RealNVPLLogpOp(realnvp, z_mean_norm, y_scale_factor)

            with pm.Model() as model:
                #prior uniforme con logistica e detrministic in modo da favorire l'hmc
                y_logit = pm.Logistic('y_logit', mu=y_init_logit, s=1, shape=(NUM_CLASSES,))
                label_vector = pm.Deterministic('label_vector', 0.55 * pm.math.sigmoid(y_logit))

                pm.Potential("likelihood", logp_op(label_vector))
                
                # 4. Sampling allineato al file singolo
                trace = pm.sample(
                    draws=DRAWS, 
                    tune=TUNE, 
                    chains=CHAINS, 
                    cores=CORES, 
                    init='adapt_diag',
                    initvals=start_vals,              # CAMBIA da 'start' a 'initvals' se non vuoi il warning (versioni future avranno initvals)
                    step=pm.NUTS(
                    target_accept=TARGET_ACCEPT, # piu alto è più forza passi piccoli
                    step_scale=STEP_SCALE,       # Suggerimento iniziale: "parti piano"
                    max_treedepth=12),       # Aggiunto target_accept esplicito
                    progressbar=False,             # Consigliato False nei loop batch
                    discard_tuned_samples=True
                )
            
            # --- CALCOLO STATISTICHE E FORMATTAZIONE ---
            post_samples = trace.posterior['label_vector'].values.reshape(-1, NUM_CLASSES)
            
            mean_est = np.mean(post_samples, axis=0)
            std_est = np.std(post_samples, axis=0)
            mode_est = calculate_mode_vector(post_samples, lower=0.0, upper=0.55)
            
            true_cls = np.argmax(y_true) + 1
            pred_cls_mean = np.argmax(mean_est) + 1
            pred_cls_mode = np.argmax(mode_est) + 1

            y_true_2d = y_true.reshape(1, -1)
            mean_2d, mode_2d = mean_est.reshape(1, -1), mode_est.reshape(1, -1)
            elapsed = time.time() - start_t

            try: 
                rhat_val = az.rhat(trace)['label_vector'].max().values
                rhat = float(rhat_val)
            except: 
                rhat = 0.0

            row_data = {
                "Index": idx,
                "True_Class": true_cls,
                "Pred_Class_Mean": pred_cls_mean,
                "Pred_Class_Mode": pred_cls_mode,
                "True_Vector": str(y_true.tolist()),
                "Mean_Vector": str(mean_est.tolist()),
                "Mode_Vector": str(mode_est.tolist()),
                "Std_Vector": str(std_est.tolist()),
                "VAE_Init": str(y_pred_init.tolist()),
                "Time": elapsed,
                "R_hat": rhat,
                
                # Metriche MEAN
                "MAE_Mean": mean_absolute_error(y_true, mean_est),
                "MSE_Mean": mean_squared_error(y_true, mean_est),
                "RMSE_Mean": np.sqrt(mean_squared_error(y_true, mean_est)),
                "R2_Mean": r2_score(y_true, mean_est),
                "CosSim_Mean": cosine_similarity(y_true_2d, mean_2d)[0][0],
                "SMAPE_Mean": smape(y_true, mean_est),
                
                # Metriche MODE
                "MAE_Mode": mean_absolute_error(y_true, mode_est),
                "MSE_Mode": mean_squared_error(y_true, mode_est),
                "RMSE_Mode": np.sqrt(mean_squared_error(y_true, mode_est)),
                "R2_Mode": r2_score(y_true, mode_est),
                "CosSim_Mode": cosine_similarity(y_true_2d, mode_2d)[0][0],
                "SMAPE_Mode": smape(y_true, mode_est),
            }
            results_list.append(row_data)
            
            print(f" Done ({elapsed:.1f}s) | True: {true_cls} | Pred: {pred_cls_mean} | R-hat: {rhat:.3f}")
            if (i+1) % 1 == 0:
                pd.DataFrame(results_list).to_csv(csv_file, index=False)
                
        except Exception as e:
            print(f" ERR: {e}")
            
    if results_list:
        pd.DataFrame(results_list).to_csv(csv_file, index=False)
        print(f"\n[DONE] NUTS completato. CSV salvato in:\n{csv_file}")
    else:
        print("Nessun risultato.")