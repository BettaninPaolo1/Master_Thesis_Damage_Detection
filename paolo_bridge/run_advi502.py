import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import tensorflow_probability as tfp
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CONFIGURAZIONE BRIDGE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")

ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_bridge.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_bridge.keras") 
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "realnvp_bridge_weights.weights.h5")

PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_params_bridge.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor_bridge.npy")

# Cartella Output
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_advi_bridge_stats")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

CSV_FILENAME = "results_batch_data_advi.csv"
CSV_PATH = os.path.join(RESULTS_DIR, CSV_FILENAME)

from data_loader_bridge import load_and_process_data, DATA_DIR_TEST, CHANNELS, SEQ_LEN, N_CLASS

# Configurazione Classi
LATENT_DIM = 4
NUM_CLASSES = N_CLASS 

# Parametri Architettura (14 Layers)
NUM_COUPLING_LAYERS = 6
HIDDEN_DIM = 64
ACTIVATION = "relu" 

# Tuning ADVI Full-Rank
NUM_TEST_SAMPLES = 200  # Campioni da testare
ITERATIONS = 500       # Iterazioni ottimizzazione ELBO
LIKELIHOOD_SCALE = 8.0  # Scala aumentata per fidarsi del RealNVP
DRAWS = 1000            # Campioni estratti dalla posterior
LEARNING_RATE = 0.01   

# ==========================================
# 2. DEFINIZIONE MODELLI (Bridge)
# ==========================================
try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable()
class CouplingMasked(layers.Layer):
    def __init__(self, latent_dim, num_classes, reg=0.001, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim, self.num_classes, self.reg = latent_dim, num_classes, reg
        self.label_net = keras.Sequential([layers.Dense(32, activation='relu'), layers.Dense(64, activation='relu'), layers.Dense(128, activation='relu')])
        self.t_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.t_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_out     = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg))
        self.s_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.s_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_out     = layers.Dense(latent_dim, activation="tanh", kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y):
        l_emb = self.label_net(y)
        t = self.t_out(self.t_joint(layers.Concatenate()([self.t_z_dense(x_masked), self.t_l_dense(l_emb)])))
        s = self.s_out(self.s_joint(layers.Concatenate()([self.s_z_dense(x_masked), self.s_l_dense(l_emb)])))
        return s, t
    
    def get_config(self):
        return super().get_config() | {"latent_dim": self.latent_dim, "num_classes": self.num_classes, "reg": self.reg}

@register_serializable()
class RealNVP_Full(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, activation=ACTIVATION, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers, self.latent_dim = num_coupling_layers, latent_dim
        self.distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim))
        self.masks = tf.constant([[1, 1, 0, 0] if i % 2 == 0 else [0, 0, 1, 1] for i in range(num_coupling_layers)], dtype=tf.float32)
        self.layers_list = [CouplingMasked(latent_dim, num_classes) for _ in range(num_coupling_layers)]

    def call(self, x, y):
        log_det_inv, z = 0, x
        for i in range(self.num_coupling_layers):
            mask = self.masks[i]
            z_masked = z * mask
            s, t = self.layers_list[i](z_masked, y)
            z = (z * mask) + ((z * (1 - mask)) * tf.exp(s) + t) * (1 - mask)
            log_det_inv += tf.reduce_sum(s * (1 - mask), axis=-1)
        return z, log_det_inv

    @tf.function
    def value_and_gradient(self, x, y, y_scale_factor_tf):
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        if len(x.shape) == 1: x = tf.expand_dims(x, 0)
        if len(y.shape) == 1: y = tf.expand_dims(y, 0)
        with tf.GradientTape() as tape:
            tape.watch(y)
            z_pred, log_det_inv = self(x, y * y_scale_factor_tf, training=False)
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            target = tf.reduce_sum(log_prob) * LIKELIHOOD_SCALE
        grad = tape.gradient(target, y)
        return target, tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
    
    def get_config(self):
        return super().get_config() | {"num_coupling_layers": self.num_coupling_layers, "latent_dim": self.latent_dim}

# ==========================================
# 3. OPS PYTENSOR (Bridge PyMC -> TF)
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
        logp_val, _ = self.model.value_and_gradient(self.x_fixed, inputs[0], self.y_scale_factor)
        outputs[0][0] = np.array(logp_val.numpy(), dtype=np.float64)
    def grad(self, inputs, output_gradients): return [output_gradients[0] * self.grad_op(inputs[0])]

# ==========================================
# 4. HELPER FUNCTIONS (STATISTICHE)
# ==========================================
def calculate_mode_vector(particles, lower=0.0, upper=0.55):
    """Calcola la moda usando KDE su ogni dimensione."""
    modes, x_grid = [], np.linspace(lower, upper, 100)
    for i in range(particles.shape[1]):
        try: modes.append(x_grid[np.argmax(gaussian_kde(particles[:, i])(x_grid))])
        except: modes.append(np.mean(particles[:, i]))
    return np.array(modes)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100 * np.mean(np.abs(y_true - y_pred) / np.where(denom==0, 1.0, denom))

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- 1. Caricamento Dati Bridge ---")
    X, y = load_and_process_data(DATA_DIR_TEST, is_training=False)
    
    z_params = np.load(PARAMS_Z_PATH)
    z_mean, z_std = z_params['mean'], z_params['std']
    y_scale = float(np.load(PARAMS_Y_PATH))

    try: keras.utils.get_custom_objects().update({"CouplingMasked": CouplingMasked, "RealNVP_Full": RealNVP_Full})
    except: pass

    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, i): return i[0] + tf.exp(0.5 * i[1]) * tf.random.normal(tf.shape(i[0]))

    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    classifier = keras.models.load_model(CLASSIFIER_PATH)
    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM)
    _ = realnvp(tf.zeros((1, 4)), tf.zeros((1, 6)))
    realnvp.load_weights(WEIGHTS_PATH)

    print(f"\n[EXEC] Inizio ADVI Loop su {NUM_TEST_SAMPLES} campioni...")
    np.random.seed(23)
    indices = np.random.choice(len(X), min(NUM_TEST_SAMPLES, len(X)), replace=False)
    results_list = []
    
    start_time = time.time()
    
    for i, idx in enumerate(indices):
        try:
            X_s = X[idx:idx+1]
            y_true = y[idx]
            true_cls = np.argmax(y_true) + 1
            
            # Encoding & Init
            z_raw, _, _ = encoder.predict(X_s, verbose=0)
            z_norm = (z_raw - z_mean)/z_std
            y_init = np.clip(classifier.predict(z_raw, verbose=0)[0], 0.001, 0.54)
            
            # PyMC ADVI
            logp_op = RealNVPLLogpOp(realnvp, z_norm, y_scale)
            with pm.Model() as model:
                label_vector = pm.Uniform('label_vector', lower=0.0, upper=0.55, shape=(NUM_CLASSES,))
                pm.Potential("likelihood", logp_op(label_vector))
                approx = pm.fit(n=ITERATIONS, method='fullrank_advi', start={'label_vector': y_init}, 
                                progressbar=False, obj_optimizer=pm.adam(learning_rate=LEARNING_RATE))
                trace = approx.sample(draws=DRAWS)
            
            # Estrazione Posterior
            post_samples = trace.posterior['label_vector'].values.reshape(-1, NUM_CLASSES)
            
            # --- CALCOLO STATISTICHE (Identiche a SVGD) ---
            mean_est = np.mean(post_samples, axis=0)
            mode_est = calculate_mode_vector(post_samples)
            std_est = np.std(post_samples, axis=0)

            true_cls = np.argmax(y_true)+1
            pred_cls = np.argmax(mean_est)+1
            
            y_true_2d = y_true.reshape(1, -1)
            mean_2d, mode_2d = mean_est.reshape(1, -1), mode_est.reshape(1, -1)
            
            row_data = {
                "Index": idx,
                "True_Class": true_cls,
                "Pred_Class_Mean": np.argmax(mean_est) + 1,
                "Pred_Class_Mode": np.argmax(mode_est) + 1,
                "True_Vector": str(y_true.tolist()),
                "Mean_Vector": str(mean_est.tolist()),
                "Mode_Vector": str(mode_est.tolist()),
                "Std_Vector": str(std_est.tolist()),
                "VAE_Init": str(y_init.tolist()),
                
                # Metriche Mean
                "MAE_Mean": mean_absolute_error(y_true, mean_est),
                "MSE_Mean": mean_squared_error(y_true, mean_est),
                "RMSE_Mean": np.sqrt(mean_squared_error(y_true, mean_est)),
                "R2_Mean": r2_score(y_true, mean_est),
                "CosSim_Mean": cosine_similarity(y_true_2d, mean_2d)[0][0],
                "SMAPE_Mean": smape(y_true, mean_est),
                
                # Metriche Mode
                "MAE_Mode": mean_absolute_error(y_true, mode_est),
                "MSE_Mode": mean_squared_error(y_true, mode_est),
                "RMSE_Mode": np.sqrt(mean_squared_error(y_true, mode_est)),
                "R2_Mode": r2_score(y_true, mode_est),
                "CosSim_Mode": cosine_similarity(y_true_2d, mode_2d)[0][0],
                "SMAPE_Mode": smape(y_true, mode_est),
            }
            results_list.append(row_data)
            
            if (i+1) % 20 == 0:
                pd.DataFrame(results_list).to_csv(CSV_PATH, index=False)
            
            print(f"{i+1:4d} | IDX: {idx:4d} | Pred: {pred_cls} | True: {true_cls} ")
                
        except Exception as e: print(f"Error {idx}: {e}")

    if results_list:
        pd.DataFrame(results_list).to_csv(CSV_PATH, index=False)
        print(f"\n[DONE] ADVI completato. CSV salvato in:\n{CSV_PATH}")
    else:
        print("Errore: Nessun risultato generato.")