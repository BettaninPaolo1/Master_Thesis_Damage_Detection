import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==========================================
# 0. CONFIGURAZIONE
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_bridge.keras")
from data_loader_bridge import load_and_process_data, DATA_DIR_TRAIN, CHANNELS, SEQ_LEN, N_CLASS

# --- PARAMETRI "ANTI-OVERFITTING" (Per curve dolci) ---
LATENT_DIM = 4          
NUM_CLASSES = N_CLASS   
NUM_COUPLING_LAYERS = 14     # <--- RIDOTTO (Da 14 a 6): Meno potenza di compressione
HIDDEN_DIM = 64             
ACTIVATION = "relu"         
REGULARIZATION = 1e-2      # <--- ALZATO (Da 1e-4 a 1e-3): Freno a mano tirato
DROPOUT_RATE = 0.1      
BATCH_SIZE = 64
EPOCHS = 300            
LEARNING_RATE = 1e-3   
LABEL_NOISE_STD = 0.001     # <--- ALZATO (Da 0.001 a 0.05): Sfocatura essenziale

SEED = 789
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================
# 1. DEFINIZIONE LAYER (STANDARD TANH)
# ==========================================

try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable()
class CouplingMasked(layers.Layer):
    def __init__(self, latent_dim, num_classes, hidden_dim=64, reg=1e-3, activation="relu", dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.reg = reg
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Reti Neurali
        self.label_net = keras.Sequential([
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(reg))
        ], name="label_emb_net")

        # Ramo T
        self.t_z_dense = layers.Dense(hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.t_l_dense = layers.Dense(hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.t_joint   = layers.Dense(32, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.t_out     = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg))
        self.t_dropout = layers.Dropout(dropout_rate)

        # Ramo S
        self.s_z_dense = layers.Dense(hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.s_l_dense = layers.Dense(hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.s_joint   = layers.Dense(32, activation=activation, kernel_regularizer=regularizers.l2(reg))
        self.s_out     = layers.Dense(latent_dim, activation="tanh", kernel_regularizer=regularizers.l2(reg)) # Tanh Standard
        self.s_dropout = layers.Dropout(dropout_rate)

    def call(self, x_masked, y, training=False):
        label_emb = self.label_net(y)
        
        # T
        t_z = self.t_dropout(self.t_z_dense(x_masked), training=training)
        t_l = self.t_l_dense(label_emb)
        t = self.t_out(self.t_joint(layers.Concatenate()([t_z, t_l])))

        # S
        s_z = self.s_dropout(self.s_z_dense(x_masked), training=training)
        s_l = self.s_l_dense(label_emb)
        s = self.s_out(self.s_joint(layers.Concatenate()([s_z, s_l])))
        
        return s, t
    
    def get_config(self):
        return super().get_config() | {
            "latent_dim": self.latent_dim, "num_classes": self.num_classes, 
            "hidden_dim": self.hidden_dim, "reg": self.reg,
            "activation": self.activation, "dropout_rate": self.dropout_rate
        }

# ==========================================
# 2. MODELLO REALNVP
# ==========================================

@register_serializable()
class RealNVP_Bridge(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, reg=1e-3, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.reg = reg
        
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        
        masks = [[1,1,0,0] if i%2==0 else [0,0,1,1] for i in range(num_coupling_layers)]
        self.masks = tf.constant(masks, dtype=tf.float32)
        
        self.layers_list = [
            CouplingMasked(latent_dim, num_classes, hidden_dim, reg, activation, DROPOUT_RATE) 
            for _ in range(num_coupling_layers)
        ]
        
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self): return [self.loss_tracker]

    def call(self, x, y, training=False):
        log_det_inv = 0
        z = x
        for i in range(self.num_coupling_layers):
            mask = self.masks[i]
            reversed_mask = 1 - mask
            z_masked = z * mask
            
            s, t = self.layers_list[i](z_masked, y, training=training)
            
            transformed = (z * reversed_mask) * tf.exp(s) + t
            z = (z * mask) + (transformed * reversed_mask)
            log_det_inv += tf.reduce_sum(s * reversed_mask, axis=-1)
        return z, log_det_inv

    def log_loss(self, x, y, training=True):
        z_base, log_det_inv = self(x, y, training=training)
        log_prob = self.distribution.log_prob(z_base)
        return -tf.reduce_mean(log_prob + log_det_inv)

    def train_step(self, data):
        x, y = data
        
        # Noise Aumentato (0.05) per evitare singolarità
        y_noisy = y + tf.random.normal(tf.shape(y), stddev=LABEL_NOISE_STD)
        
        with tf.GradientTape() as tape:
            log_lik_loss = self.log_loss(x, y_noisy, training=True)
            reg_loss = tf.reduce_sum(self.losses)
            total_loss = log_lik_loss + reg_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.loss_tracker.update_state(total_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        loss = self.log_loss(x, y, training=False) + tf.reduce_sum(self.losses)
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}
        
    def get_config(self):
        return super().get_config() | {
            "num_coupling_layers": self.num_coupling_layers, "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim, "reg": self.reg
        }

# ==========================================
# 3. MAIN
# ==========================================

if __name__ == "__main__":
    print(f"--- TRAINING BRIDGE ANTI-OVERFITTING (6 Layers, High Noise) ---")
    
    # 1. Load Data & Encoder
    X_train_raw, y_train_raw = load_and_process_data(DATA_DIR_TRAIN, is_training=False, added_snr=100)
    max_y = np.max(y_train_raw); scale_factor = 1.0/max_y if max_y>0 else 1.0; y_scaled = y_train_raw * scale_factor
    np.save(os.path.join(OUTPUT_DIR, "y_scale_factor_bridge.npy"), scale_factor)

    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, i): return i[0] + tf.exp(0.5 * i[1]) * tf.random.normal(tf.shape(i[0]))

    if not os.path.exists(ENCODER_PATH): raise FileNotFoundError(f"Manca Encoder: {ENCODER_PATH}")
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling}); encoder.trainable = False
    
    z_raw, _, _ = encoder.predict(X_train_raw, batch_size=64, verbose=1)
    z_mean, z_std = np.mean(z_raw, axis=0), np.std(z_raw, axis=0); z_std[z_std==0]=1.0
    Z_norm = (z_raw - z_mean) / z_std
    np.savez(os.path.join(OUTPUT_DIR, "z_params_bridge.npz"), mean=z_mean, std=z_std)
    
    Z_train, Z_val, y_train, y_val = train_test_split(Z_norm, y_scaled, test_size=0.2, random_state=42)

    # 2. Train
    realnvp = RealNVP_Bridge(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, REGULARIZATION, ACTIVATION)
    _ = realnvp(Z_train[:1], y_train[:1]) 
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(LEARNING_RATE, decay_steps=EPOCHS*len(Z_train)//BATCH_SIZE)
    realnvp.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
    
    history = realnvp.fit(
        Z_train, y_train, validation_data=(Z_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=20, restore_best_weights=True)],
        verbose=1
    )
    
    realnvp.save_weights(os.path.join(OUTPUT_DIR, "realnvp_bridge_weights.weights.h5"))
    print("\n✅ Training Finito. Pesi salvati.")