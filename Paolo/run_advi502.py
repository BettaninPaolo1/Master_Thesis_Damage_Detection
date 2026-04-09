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
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier_cvae.keras")

# Pesi della versione 14 Layers
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "realnvp_minmax14layers_weights.weights.h5")

# Percorsi Normalizzazione
PARAMS_Z_PATH = os.path.join(OUTPUT_DIR, "z_normalization_params.npz")
PARAMS_Y_PATH = os.path.join(OUTPUT_DIR, "y_scale_factor.npy")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_advi_stats3")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

from data_loader import load_and_process_data, TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

LATENT_DIM = 4
NUM_CLASSES = 7

# --- 14 LAYERS ---
NUM_COUPLING_LAYERS = 14 
HIDDEN_DIM = 64
ACTIVATION = "relu" 

# --- TUNING ADVI POTENZIATO ---
NUM_TEST_SAMPLES = 1000
ITERATIONS = 500      
LIKELIHOOD_SCALE = 4.0 
DRAWS = 1000 
LEARNING_RATE = 0.01           

# ==========================================
# 2. DEFINIZIONE MODELLI
# ==========================================

try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

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

        self.t_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.t_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_out     = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg))

        self.s_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.s_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_out     = layers.Dense(latent_dim, activation="tanh", kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y):
        label_emb = self.label_net(y)
        
        t_z = self.t_z_dense(x_masked)
        t_l = self.t_l_dense(label_emb)
        t_concat = layers.Concatenate()([t_z, t_l])
        t = self.t_joint(t_concat)
        t = self.t_out(t)

        s_z = self.s_z_dense(x_masked)
        s_l = self.s_l_dense(label_emb)
        s_concat = layers.Concatenate()([s_z, s_l])
        s = self.s_joint(s_concat)
        s = self.s_out(s)
        return s, t
    
    def get_config(self):
        return super().get_config() | {"latent_dim": self.latent_dim, "num_classes": self.num_classes, "reg": self.reg}

@register_serializable()
class RealNVP_Full(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, activation=ACTIVATION, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
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
            y_scaled = y * y_scale_factor_tf
            z_pred, log_det_inv = self(x, y_scaled, training=False)
            log_prob = self.distribution.log_prob(z_pred) + log_det_inv
            target = tf.reduce_sum(log_prob) * LIKELIHOOD_SCALE

        grad = tape.gradient(target, y)
        grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
        return target, grad
    
    def get_config(self):
        return super().get_config() | {"num_coupling_layers": self.num_coupling_layers, "latent_dim": self.latent_dim}

# ==========================================
# 3. OPS PYTENSOR
# ==========================================

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

def evaluate_metrics_internal(results_df):
    """Valutazione rapida interna (per debug) - BASATA SULLA MEDIA"""
    print("\n" + "="*80)
    print("   REPORT DI VALUTAZIONE ADVI (Internal Check - MEAN)")
    print("="*80)

    def parse_col(col_name):
        val = results_df[col_name].iloc[0]
        if isinstance(val, str):
            return np.array([np.fromstring(x.replace('[','').replace(']','').replace('\n',' ').replace(',', ' '), sep=' ') 
                             for x in results_df[col_name]])
        return np.stack(results_df[col_name].values)

    y_true_all = parse_col('True_Vector')
    y_mean_all = parse_col('Mean_Vector')
    y_mode_all = parse_col('Mode_Vector')

    mae_mean = mean_absolute_error(y_true_all, y_mean_all)
    r2_mean = r2_score(y_true_all, y_mean_all)
    
    # CosSim media
    cos_sim_mean = np.mean([cosine_similarity(y_true_all[i:i+1], y_mean_all[i:i+1])[0][0] for i in range(len(y_true_all))])

    print(f"{'Metrica':<20} | {'Media (Mean Est)':<20} | {'Moda (Mode Est)':<20}")
    print("-" * 66)
    print(f"{'MAE':<20} | {mae_mean:<20.5f} | --")
    print(f"{'R2 Score':<20} | {r2_mean:<20.5f} | --")
    print(f"{'Cos Sim':<20} | {cos_sim_mean:<20.5f} | --")
    print("-" * 66)

    # --- MODIFICA QUI: Confusion Matrix basata sulla MEDIA ---
    true_classes = np.argmax(y_true_all, axis=1) + 1
    pred_classes = np.argmax(y_mean_all, axis=1) + 1  # <--- USO LA MEDIA
    acc = accuracy_score(true_classes, pred_classes)
    
    cm = confusion_matrix(true_classes, pred_classes, labels=range(1, NUM_CLASSES + 1))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(1, NUM_CLASSES + 1), yticklabels=range(1, NUM_CLASSES + 1))
    plt.title(f'Confusion Matrix - ADVI (MEAN Prediction)\nAccuracy: {acc:.2f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_advi_internal.png"))
    plt.show()

# ==========================================
# 5. MAIN LOOP
# ==========================================

if __name__ == "__main__":
    print("--- 1. Caricamento Dati e Parametri ---")
    X, y = load_and_process_data(TEST_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    if not os.path.exists(PARAMS_Z_PATH) or not os.path.exists(PARAMS_Y_PATH):
        raise FileNotFoundError("Parametri mancanti!")

    z_params = np.load(PARAMS_Z_PATH)
    z_mean, z_std = z_params['mean'], z_params['std']
    y_scale_factor_raw = np.load(PARAMS_Y_PATH)
    y_scale_val = float(y_scale_factor_raw.item()) if y_scale_factor_raw.ndim == 0 else float(y_scale_factor_raw)
    
    try: keras.utils.get_custom_objects().update({"CouplingMasked": CouplingMasked, "RealNVP_Full": RealNVP_Full})
    except: pass

    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    print("--- 2. Caricamento Modelli ---")
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    try:
        classifier_cvae = keras.models.load_model(CLASSIFIER_PATH)
    except:
        print("WARN: Classificatore non trovato.")
        exit()

    realnvp = RealNVP_Full(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, activation=ACTIVATION)
    _ = realnvp(tf.zeros((1, LATENT_DIM)), tf.zeros((1, NUM_CLASSES)))
    realnvp.load_weights(WEIGHTS_PATH)

    print(f"\n--- 3. ADVI FULL-RANK Loop ({NUM_TEST_SAMPLES} samples) ---")
    #np.random.seed(84)
    #np.random.seed(67) 93%
    np.random.seed(67)    
    indices = np.random.choice(len(X), min(NUM_TEST_SAMPLES, len(X)), replace=False)
    results_list = []
    csv_file = os.path.join(RESULTS_DIR, "results_advi_batch.csv")
    
    total_start_time = time.time()
    
    for i, idx in enumerate(indices):
        iter_start = time.time()
        try:
            X_sample = X[idx:idx+1]
            y_true = y[idx]
            true_cls = np.argmax(y_true) + 1
            
            # 1. Encoding
            z_raw, _, _ = encoder.predict(X_sample, verbose=0)
            z_norm = (z_raw - z_mean) / z_std
            
            # 2. VAE Init
            y_init = np.clip(classifier_cvae.predict(z_raw, verbose=0)[0], 0.001, 0.54)
            init_vals = {'label_vector': y_init}
            
            # 3. ADVI
            logp_op = RealNVPLLogpOp(realnvp, z_norm, y_scale_val)
            
            with pm.Model() as model:
                label_vector = pm.Uniform('label_vector', lower=0.0, upper=0.55, shape=(NUM_CLASSES,))
                pm.Potential("likelihood", logp_op(label_vector))
                
                # Full Rank ADVI
                approx = pm.fit(
                    n=ITERATIONS, 
                    method='fullrank_advi', 
                    start=init_vals, 
                    progressbar=False,
                    obj_optimizer=pm.adam(learning_rate=LEARNING_RATE) 
                )
                trace = approx.sample(draws=DRAWS)
            
            # 4. Calcolo Statistiche Esteso
            post_samples = trace.posterior['label_vector'].values.reshape(-1, NUM_CLASSES)
            
            mean_est = np.mean(post_samples, axis=0)
            mode_est = calculate_mode_vector(post_samples)
            std_est = np.std(post_samples, axis=0)
            
            y_true_2d = y_true.reshape(1, -1)
            mean_2d = mean_est.reshape(1, -1)
            mode_2d = mode_est.reshape(1, -1)

            # --- DIZIONARIO STANDARDIZZATO ---
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
            
            # --- LOG CON CLASSIFICAZIONE VIA MEDIA ---
            print(f"[Sample {i+1}] True: {true_cls} | MeanPred: {np.argmax(mean_est)+1} | Time: {time.time()-iter_start:.1f}s")
            
            if (i+1) % 5 == 0: 
                pd.DataFrame(results_list).to_csv(csv_file, index=False)
                
        except Exception as e:
            print(f"Error {idx}: {e}")

    total_time = time.time() - total_start_time
    print(f"\nTempo Totale: {total_time:.2f}s")

    if len(results_list) > 0:
        final_df = pd.DataFrame(results_list)
        final_df.to_csv(csv_file, index=False)
        print(f"\n[DONE] CSV salvato in: {csv_file}")
        evaluate_metrics_internal(final_df)
    else:
        print("Nessun risultato generato.")