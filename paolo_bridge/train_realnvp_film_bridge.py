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

# --- PARAMETRI ARCHITETTURA (Bridge 4D + Deep FiLM 3L) ---
LATENT_DIM = 4          
NUM_CLASSES = N_CLASS   
NUM_COUPLING_LAYERS = 14     
HIDDEN_DIM = 128            
ACTIVATION = "relu"         
REGULARIZATION = 1e-3    
BATCH_SIZE = 64
EPOCHS = 250            
LEARNING_RATE = 5e-4  
LABEL_NOISE_STD = 0.001     

SEED = 789
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================
# 1. ARCHITETTURA DEEP FiLM (3 LAYERS + RESIDUAL)
# ==========================================

try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

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
        
        # Embedding Globale - RIGOROSAMENTE LINEARE
        self.label_embedding = keras.Sequential([
            layers.Dense(64, activation='swish', kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(128, activation='swish', kernel_regularizer=regularizers.l2(reg)),
            layers.Dense(128, activation='linear', kernel_regularizer=regularizers.l2(reg)) 
        ], name="global_label_emb")

        # --- RAMO T (Traslazione) - 3 LAYERS ---
        self.t_layer1 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.t_layer2 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.t_layer3 = FiLMDense(hidden_dim, activation='relu', reg=reg) 
        
        self.t_out = layers.Dense(latent_dim, activation='linear', 
                                  kernel_initializer='zeros', 
                                  kernel_regularizer=regularizers.l2(reg))

        # --- RAMO S (Scalatura) - 3 LAYERS ---
        self.s_layer1 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.s_layer2 = FiLMDense(hidden_dim, activation='relu', reg=reg)
        self.s_layer3 = FiLMDense(hidden_dim, activation='relu', reg=reg) 
        
        self.s_out = layers.Dense(latent_dim, activation='tanh', 
                                  kernel_initializer='zeros',
                                  kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y):
        y_emb = self.label_embedding(y)

        # --- RAMO T (Con Doppia Residual Connection) ---
        t = self.t_layer1(x_masked, y_emb)      
        t_res1 = self.t_layer2(t, y_emb)        
        t = t + t_res1                          
        t_res2 = self.t_layer3(t, y_emb)        
        t = t + t_res2                          
        t_final = self.t_out(t)

        # --- RAMO S (Con Doppia Residual Connection) ---
        s = self.s_layer1(x_masked, y_emb)      
        s_res1 = self.s_layer2(s, y_emb)        
        s = s + s_res1                          
        s_res2 = self.s_layer3(s, y_emb)        
        s = s + s_res2                          
        s_final = self.s_out(s)

        return s_final, t_final

    def get_config(self):
        return super().get_config() | {"latent_dim": self.latent_dim, "hidden_dim": self.hidden_dim, "reg": self.reg}

@register_serializable()
class RealNVP_Bridge_3L(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.reg = reg
        
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim)
        )
        
        # Maschere per spazio 4D (metà 1 e metà 0 alternate)
        masks = []
        for i in range(num_coupling_layers):
            mask = np.zeros(latent_dim, dtype=np.float32)
            if i % 2 == 0: mask[:latent_dim//2] = 1
            else: mask[latent_dim//2:] = 1
            masks.append(mask)
        self.masks = tf.constant(masks, dtype=tf.float32)
        
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
        log_prob = self.distribution.log_prob(z_base)
        return -tf.reduce_mean(log_prob + log_det_inv)

    def train_step(self, data):
        x, y = data
        noise = tf.random.normal(shape=tf.shape(y), stddev=LABEL_NOISE_STD)
        y_noisy = tf.maximum(y + noise, 0.0) 
        
        with tf.GradientTape() as tape:
            log_lik_loss = self.log_loss(x, y_noisy)
            reg_loss = tf.reduce_sum(self.losses)
            total_loss = log_lik_loss + reg_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        loss = self.log_loss(x, y) + tf.reduce_sum(self.losses)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        
    def get_config(self):
        return super().get_config() | {
            "num_coupling_layers": self.num_coupling_layers, "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim, "reg": self.reg
        }

class TrainingMonitorCallback(keras.callbacks.Callback):
    def __init__(self, sample_x, sample_y, print_every_n_epochs=10):
        super().__init__()
        self.sample_x = tf.convert_to_tensor(sample_x, dtype=tf.float32)
        self.sample_y = tf.convert_to_tensor(sample_y, dtype=tf.float32)
        self.print_every_n_epochs = print_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Stampa base ad ogni epoca (opzionale se usi verbose=1 nel fit, ma utile per formattazione)
        print(f"\n--- Epoch {epoch + 1} Fine ---")
        print(f"Loss: {logs.get('loss', 0):.4f} | Val Loss: {logs.get('val_loss', 0):.4f}")

        # Controlli più profondi ogni N epoche
        if (epoch + 1) % self.print_every_n_epochs == 0:
            print(f"\n[Monitoraggio Dettagliato Epoca {epoch + 1}]")
            
            # 1. Controlla i pesi (es. della prima layer S)
            first_layer = self.model.layers_list[0]
            s_weights = first_layer.s_out.get_weights()
            if s_weights:
                w = s_weights[0]
                print(f"  -> Livello 0 's_out' pesi: Media = {np.mean(w):.6f}, Std = {np.std(w):.6f}, Max = {np.max(np.abs(w)):.6f}")
                
            # 2. Guarda l'output effettivo S e T per un piccolo batch di validazione
            mask = self.model.masks[0]
            z_masked = self.sample_x * mask
            
            s, t = first_layer(z_masked, self.sample_y)
            
            print(f"  -> S (Scaling) stats: Media = {tf.reduce_mean(s):.6f}, Min = {tf.reduce_min(s):.6f}, Max = {tf.reduce_max(s):.6f}")
            print(f"  -> T (Translation) stats: Media = {tf.reduce_mean(t):.6f}, Min = {tf.reduce_min(t):.6f}, Max = {tf.reduce_max(t):.6f}")
            
            # 3. Controlla se la loss calcolata manualmente ha senso
            manual_loss = self.model.log_loss(self.sample_x, self.sample_y)
            print(f"  -> Loss sul sample batch: {manual_loss:.4f}")
            print("-" * 50)

# ==========================================
# 3. MAIN
# ==========================================

if __name__ == "__main__":
    print(f"--- TRAINING BRIDGE: 4D, 3-Layer Deep FiLM + Residual ---")
    
    # 1. Load Data & Encoder
    X_train_raw, y_train_raw = load_and_process_data(DATA_DIR_TRAIN, is_training=True)
    max_y = np.max(y_train_raw)
    scale_factor = 1.0/max_y if max_y>0 else 1.0
    y_scaled = y_train_raw * scale_factor
    np.save(os.path.join(OUTPUT_DIR, "y_scale_factor_bridge.npy"), scale_factor)

    @register_serializable()
    class Sampling(layers.Layer):
        def call(self, i): return i[0] + tf.exp(0.5 * i[1]) * tf.random.normal(tf.shape(i[0]))

    if not os.path.exists(ENCODER_PATH): 
        raise FileNotFoundError(f"Manca Encoder: {ENCODER_PATH}")
        
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    encoder.trainable = False
    
    z_raw, _, _ = encoder.predict(X_train_raw, batch_size=64, verbose=1)
    z_mean, z_std = np.mean(z_raw, axis=0), np.std(z_raw, axis=0)
    z_std[z_std==0] = 1.0
    Z_norm = (z_raw - z_mean) / z_std
    np.savez(os.path.join(OUTPUT_DIR, "z_params_bridge.npz"), mean=z_mean, std=z_std)
    
    Z_train, Z_val, y_train, y_val = train_test_split(Z_norm, y_scaled, test_size=0.2, random_state=42)

    # 2. Train
    realnvp = RealNVP_Bridge_3L(NUM_COUPLING_LAYERS, LATENT_DIM, NUM_CLASSES, HIDDEN_DIM, REGULARIZATION)
    
    print("Costruzione modello...")
    _ = realnvp(Z_train[:1], y_train[:1]) 
    
    realnvp.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    # Prendiamo un piccolo campione fisso di dati (es. 16 campioni) per monitorare come cambiano le predizioni su di essi
    sample_batch_z = Z_val[:16]
    sample_batch_y = y_val[:16]
    
    monitor_cb = TrainingMonitorCallback(
        sample_x=sample_batch_z, 
        sample_y=sample_batch_y, 
        print_every_n_epochs=5  # Stampa i dettagli ogni 5 epoche
    )
    
    print("Inizio Addestramento")
    history = realnvp.fit(
        Z_train, y_train, 
        validation_data=(Z_val, y_val),
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, monitor_cb],
        verbose=1
    )
    
    # SALVATAGGIO CON NUOVO NOME PER NON SOVRASCRIVERE QUELLO VECCHIO
    weights_path = os.path.join(OUTPUT_DIR, "realnvp_bridge_3layer_film_weights.weights.h5")
    realnvp.save_weights(weights_path)
    print(f"\n✅ Training Finito. Pesi salvati in: {weights_path}")
    
    # Plot
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('RealNVP Bridge 4D (3-Layer FiLM + Residual) Loss')
    plt.legend()
    plt.show()