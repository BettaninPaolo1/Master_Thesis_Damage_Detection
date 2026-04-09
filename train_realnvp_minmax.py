import os
import numpy as np
import random
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
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")

from data_loader import load_and_process_data, DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# PARAMETRI ARCHITETTURA (Manteniamo i tuoi standard)
LATENT_DIM = 4          
NUM_CLASSES = 7         
NUM_COUPLING_LAYERS = 14
HIDDEN_DIM = 64         
ACTIVATION = "relu"      # FONDAMENTALE per NUTS e gradienti lisci
DROPOUT_RATE = 0.1

# PARAMETRI TRAINING "ANTI-BIAS"
BATCH_SIZE = 64
EPOCHS = 100             # Abbondiamo, c'è l'early stopping
LEARNING_RATE = 5e-4      # velocità di apprendimento, se vedi troppi salti abbassa
REGULARIZATION = 0.001   # se troppo bassa overfitting
LABEL_NOISE_STD = 0.01 # JITTERING: Aiuta a interpolare tra i livelli discreti

SEED = random.randint(0, 2**32 - 1)
#SEED = 964051881                #loss: -4.8377 - val_loss: -4.2969
#SEED = 1213881896               #loss: -4.5712 - val_loss: -4.4309
#SEED = 1778906487               
#SEED = 881576201                # questo con reg = 1e-5 e lr = 1e-4
#SEED = 3242310650               

# 2. STAMPA IL SEED (Così puoi copiartelo)
print(f"\n{'='*40}")
print(f"   SEED GENERATO PER QUESTA RUN: {SEED}")
print(f"   Salvalo se i risultati sono bueni")
print(f"{'='*40}\n")

# 3. Applica il seed a tutte le librerie per garantire che la run sia coerente
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================
# 1. DEFINIZIONE MODELLO
# ==========================================

class CouplingMasked(layers.Layer):
    def __init__(self, latent_dim, num_classes, hidden_dim=64, reg=0.01, activation="relu"):
        super().__init__()
        
        # 1. LABEL EMBEDDING (Il "Cervello" che mancava)
        self.label_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu')
        ], name="label_embedding")

        # 2. RAMO T (Traslazione) con Dropout
        self.t_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_dropout = layers.Dropout(DROPOUT_RATE) # Usa la variabile globale
        self.t_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.t_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.t_out     = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(reg))

        # 3. RAMO S (Scalatura) con Dropout
        self.s_z_dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_dropout = layers.Dropout(DROPOUT_RATE) # Usa la variabile globale
        self.s_l_dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.s_joint   = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(reg))
        self.s_out     = layers.Dense(latent_dim, activation="tanh", kernel_regularizer=regularizers.l2(reg))

    def call(self, x_masked, y, training=False):
        # A. Embedding del Label
        label_emb = self.label_net(y)

        # B. Calcolo T (Traslazione)
        t_z = self.t_z_dense(x_masked)
        t_z = self.t_dropout(t_z, training=training) # Dropout attivo solo in training
        t_l = self.t_l_dense(label_emb)
        
        # Fusione Profonda
        t_concat = layers.Concatenate()([t_z, t_l])
        t = self.t_joint(t_concat)
        t = self.t_out(t)

        # C. Calcolo S (Scalatura)
        s_z = self.s_z_dense(x_masked)
        s_z = self.s_dropout(s_z, training=training) # Dropout attivo solo in training
        s_l = self.s_l_dense(label_emb)
        
        # Fusione Profonda
        s_concat = layers.Concatenate()([s_z, s_l])
        s = self.s_joint(s_concat)
        s = self.s_out(s)

        return s, t

class RealNVP_BiasFix(keras.Model):
    def __init__(self, num_coupling_layers, latent_dim, num_classes, hidden_dim, reg=0.01, activation=ACTIVATION):
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
        
        self.layers_list = [
            CouplingMasked(latent_dim, num_classes, hidden_dim, reg, activation) 
            for _ in range(num_coupling_layers)
        ]
        
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

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

    def log_loss(self, x, y):
        z_base, log_det_inv = self(x, y)
        log_prob_base = self.distribution.log_prob(z_base)
        log_likelihood = log_prob_base + log_det_inv
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        x, y = data
        
        # === TRUCCO ANTI-BIAS: Label Jittering ===
        # Aggiungiamo un rumore infinitesimale alle etichette.
        # Questo costringe la rete a generalizzare e centrare meglio la predizione.
        noise = tf.random.normal(shape=tf.shape(y), mean=0.0, stddev=LABEL_NOISE_STD)
        y_noisy = y + noise
        # Clipping opzionale: se i danni fisici non possono essere < 0
        y_noisy = tf.maximum(y_noisy, 0.0) 
        
        with tf.GradientTape() as tape:
            # Calcoliamo la loss usando le label "sporcate"
            loss = self.log_loss(x, y_noisy)
            
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        x, y = data
        # In validazione usiamo le label pulite (Ground Truth)
        loss = self.log_loss(x, y)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

# ==========================================
# 2. ESECUZIONE
# ==========================================

if __name__ == "__main__":
    print("--- Training RealNVP con Bias Correction ---")
    
    # A. Caricamento Dati
    X, y = load_and_process_data(DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    # Scaling Y
    max_val = np.max(y)
    scale_factor = 1.0 / max_val if max_val > 0 else 1.0
    y_scaled = y * scale_factor
    print(f"Scala Y applicata: {scale_factor:.4f}")
    np.save(os.path.join(OUTPUT_DIR, "y_scale_factor.npy"), scale_factor)

    # B. Estrazione Z dal CVAE (Congelato)
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("Encoder non trovato.")
        
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    print("Caricamento Encoder...")
    encoder = keras.models.load_model(ENCODER_PATH, custom_objects={"Sampling": Sampling})
    encoder.trainable = False # NON lo tocchiamo
    
    print("Proiezione nello Spazio Latente...")
    z_dataset_mean, _, _ = encoder.predict(X, batch_size=64, verbose=1)
    
    # Normalizzazione Z Standard (Media 0, Std 1)
    # Fondamentale perché RealNVP assume una normale standard alla base
    z_mean = np.mean(z_dataset_mean, axis=0)
    z_std = np.std(z_dataset_mean, axis=0)
    z_std[z_std == 0] = 1.0 
    z_dataset_mean = (z_dataset_mean - z_mean) / z_std
    
    # Salviamo i parametri per usarli in inferenza
    np.savez(os.path.join(OUTPUT_DIR, "z_normalization_params.npz"), mean=z_mean, std=z_std)
    
    # Split
    Z_train, Z_val, y_train, y_val = train_test_split(z_dataset_mean, y_scaled, test_size=0.2, random_state=42)

    # C. Training
    print(f"Configurazione: Reg={REGULARIZATION}, Noise={LABEL_NOISE_STD}, Act={ACTIVATION}")
    
    realnvp = RealNVP_BiasFix(
        num_coupling_layers=NUM_COUPLING_LAYERS,
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        reg=REGULARIZATION,
        activation=ACTIVATION
    )
    
    # Dummy pass
    _ = realnvp(Z_train[:1], y_train[:1])
    
    #lr_schedule = keras.optimizers.schedules.CosineDecay(
    #    initial_learning_rate=LEARNING_RATE,
    #    decay_steps=EPOCHS * (len(Z_train) // BATCH_SIZE),
    #    alpha=0.5  # Arriverà al 10% del LR iniziale alla fine
    #)

    realnvp.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    
    history = realnvp.fit(
        x=Z_train, y=y_train,
        validation_data=(Z_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Salvataggio Pesi
    weights_path = os.path.join(OUTPUT_DIR, "realnvp_minmax14layers_weights.weights.h5")
    realnvp.save_weights(weights_path)
    print(f"Training completato. Pesi salvati in: {weights_path}")
    
    # Plot Loss
    plt.plot(history.history['loss'], label='Train Loss (with Noise)')
    plt.plot(history.history['val_loss'], label='Val Loss (Clean)')
    plt.title('RealNVP Bias-Correction Training')
    plt.legend()
    plt.show()