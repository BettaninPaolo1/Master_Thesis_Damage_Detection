import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers, regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==========================================
# 0. CONFIGURAZIONE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Salviamo in una cartella distinta "v3" per indicare la versione a 3 layer
MODEL_DIR = os.path.join(SCRIPT_DIR, "models_evolution_v2_deepfilm_3layers")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "models_evolution_v2", "encoder_cvae_8d.keras")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

from data_loader import load_and_process_data, DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# PARAMETRI ARCHITETTURA
LATENT_DIM = 8          
NUM_CLASSES = 7         
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 128        
ACTIVATION = "relu"

# PARAMETRI TRAINING
BATCH_SIZE = 64
EPOCHS = 250            # Aumentiamo leggermente le epoche dato che la rete è più profonda
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-5  
LABEL_NOISE_STD = 0.001

# FIX SERIALIZZAZIONE
try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

# ==========================================
# 1. ARCHITETTURA DEEP FiLM (3 LAYERS + RESIDUAL)
# ==========================================

@register_serializable()
class FiLMDense(layers.Layer):
    """
    Blocco denso con modulazione FiLM.
    """
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
        # FiLM: (1 + gamma) * x + beta
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
        
        # Embedding Globale
        self.label_embedding = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu') 
        ], name="global_label_emb")

        # --- RAMO T (Traslazione) - 3 LAYERS ---
        self.t_layer1 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.t_layer2 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.t_layer3 = FiLMDense(hidden_dim, activation='relu', reg=reg) # <--- 3° Layer
        
        self.t_out = layers.Dense(latent_dim, activation='linear', 
                                  kernel_initializer='zeros', 
                                  kernel_regularizer=regularizers.l2(reg))

        # --- RAMO S (Scalatura) - 3 LAYERS ---
        self.s_layer1 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.s_layer2 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.s_layer3 = FiLMDense(hidden_dim, activation='relu', reg=reg) # <--- 3° Layer
        
        self.s_out = layers.Dense(latent_dim, activation='tanh', 
                                  kernel_initializer='zeros',
                                  kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y):
        y_emb = self.label_embedding(y)

        # --- RAMO T (Con Doppia Residual Connection) ---
        t = self.t_layer1(x_masked, y_emb)      # Layer 1
        
        t_res1 = self.t_layer2(t, y_emb)        # Layer 2
        t = t + t_res1                          # Skip 1
        
        t_res2 = self.t_layer3(t, y_emb)        # Layer 3
        t = t + t_res2                          # Skip 2 (Nuova riga!)
        
        t_final = self.t_out(t)

        # --- RAMO S (Con Doppia Residual Connection) ---
        s = self.s_layer1(x_masked, y_emb)      # Layer 1
        
        s_res1 = self.s_layer2(s, y_emb)        # Layer 2
        s = s + s_res1                          # Skip 1
        
        s_res2 = self.s_layer3(s, y_emb)        # Layer 3
        s = s + s_res2                          # Skip 2 (Nuova riga!)
        
        s_final = self.s_out(s)

        return s_final, t_final

    def get_config(self):
        return super().get_config() | {"latent_dim": self.latent_dim, "hidden_dim": self.hidden_dim, "reg": self.reg}

@register_serializable()
class RealNVP_8D_3L(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.reg = reg
        
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        
        masks = []
        for i in range(num_coupling_layers):
            mask = np.zeros(latent_dim, dtype=np.float32)
            if i % 2 == 0: mask[:latent_dim//2] = 1
            else: mask[latent_dim//2:] = 1
            masks.append(mask)
        self.masks = tf.constant(masks, dtype=tf.float32)
        
        # Usiamo la nuova classe a 3 Layer
        self.layers_list = [
            CouplingMaskedDeepFiLM_3L(latent_dim, num_classes, hidden_dim, reg) 
            for _ in range(num_coupling_layers)
        ]
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self): return [self.loss_tracker]

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

    def log_loss(self, x, y):
        z_base, log_det_inv = self(x, y)
        log_prob_base = self.distribution.log_prob(z_base)
        return -tf.reduce_mean(log_prob_base + log_det_inv)

    def train_step(self, data):
        x, y = data
        noise = tf.random.normal(shape=tf.shape(y), stddev=LABEL_NOISE_STD)
        y_noisy = tf.maximum(y + noise, 0.0) 
        
        with tf.GradientTape() as tape:
            # 1. Loss principale (Log-Likelihood)
            log_lik = self.log_loss(x, y_noisy)
            
            # 2. Recupera la Regolarizzazione (IL PEZZO MANCANTE)
            reg_loss = tf.reduce_sum(self.losses)
            
            # 3. Somma tutto
            total_loss = log_lik + reg_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        loss = self.log_loss(x, y)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        
    def get_config(self):
        return super().get_config() | {
            "num_coupling_layers": self.num_coupling_layers,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "reg": self.reg
        }

# ==========================================
# 2. ESECUZIONE TRAINING
# ==========================================

if __name__ == "__main__":
    print("--- Training RealNVP 8D (3-Layer Deep FiLM + Residual) ---")
    
    # A. Caricamento Dati
    X, y = load_and_process_data(DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    max_y = np.max(y)
    scale_factor = 1.0 / max_y if max_y > 0 else 1.0
    y_scaled = y * scale_factor
    np.save(os.path.join(MODEL_DIR, "y_scale_factor_8d.npy"), scale_factor)

    # B. Proiezione Latente
    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder non trovato in {ENCODER_PATH}")

    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    z_dataset_mean, _, _ = encoder.predict(X, batch_size=64, verbose=1)
    
    # Normalizzazione
    z_mean = np.mean(z_dataset_mean, axis=0)
    z_std = np.std(z_dataset_mean, axis=0)
    z_std[z_std == 0] = 1.0 
    z_dataset_norm = (z_dataset_mean - z_mean) / z_std
    
    np.savez(os.path.join(MODEL_DIR, "z_normalization_params_8d.npz"), mean=z_mean, std=z_std)
    
    Z_train, Z_val, y_train, y_val = train_test_split(z_dataset_norm, y_scaled, test_size=0.2, random_state=42)

    # C. Model Compile
    realnvp = RealNVP_8D_3L(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, REGULARIZATION)
    
    # DUMMY PASS
    print("Costruzione modello...")
    _ = realnvp(Z_train[:1], y_train[:1]) 
    
    realnvp.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    
    # Aumentiamo la pazienza perché la rete è più profonda
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    
    history = realnvp.fit(
        Z_train, y_train,
        validation_data=(Z_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop]
    )
    
    # Salvataggio
    weights_path = os.path.join(MODEL_DIR, "realnvp_8d_3layer_film_weights.weights.h5")
    realnvp.save_weights(weights_path)
    print(f"Pesi salvati in: {weights_path}")

    # Plot
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('RealNVP 8D (3-Layer FiLM + Residual) Loss')
    plt.legend()
    plt.show()