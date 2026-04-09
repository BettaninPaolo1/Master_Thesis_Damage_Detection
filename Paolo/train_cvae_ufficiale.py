import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, ops
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Genera un numero casuale 
#SEED = random.randint(0, 2**32 - 1)
#SEED = 3388967444               # val_kl_loss: 9.2240 - val_loss: 41.1403 - val_recon_loss: 24.3367
#SEED = 12327450                 # val_kl_loss: 9.2497 - val_loss: 40.9192 - val_recon_loss: 23.6952
#SEED = 1266925345               # val_kl_loss: 9.0876 - val_loss: 40.4566 - val_recon_loss: 24.5787
SEED = 964051881                # val_kl_loss: 8.8612 - val_loss: 34.7044 - val_recon_loss: 21.7188 <- vincitore

# 2. STAMPA IL SEED (Così puoi copiartelo)
print(f"\n{'='*40}")
print(f"   SEED GENERATO PER QUESTA RUN: {SEED}")
print(f"   --> Salvatelo se i risultati sono ottimi!")
print(f"{'='*40}\n")

# 3. Applica il seed a tutte le librerie per garantire che la run sia coerente
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Fissiamo i semi per rendere l'addestramento RIPRODUCIBILE
#SEED = 42
#os.environ['PYTHONHASHSEED'] = str(SEED)
#random.seed(SEED)
#np.random.seed(SEED)
#tf.random.set_seed(SEED)

# ==========================================
# 0. CONFIGURAZIONE PERCORSI
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution")

# Importiamo il loader
from data_loader import load_and_process_data, DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR
#from data_loader_confronto import load_and_process_data, DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# ==========================================
# 1. CONFIGURAZIONE IPERPARAMETRI
# ==========================================
LATENT_DIM = 4          
NUM_CLASSES = 7         
BATCH_SIZE = 32
EPOCHS = 250
LEARNING_RATE = 1e-3
KL_WEIGHT = 1.0         
CLASS_WEIGHT = 3000.0   

FILTERS = [32, 64, 32]
KERNEL_SIZES = [25, 13, 7]
STRIDES = 1
DROP_RATE = 0.05
L2_REG = 1e-3

# ==========================================
# 2. DEFINIZIONE DEL MODELLO
# ==========================================

class Sampling(layers.Layer):
    """Usa (z_mean, z_log_var) per campionare z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_shape):
    encoder_inputs = layers.Input(shape=input_shape, name="encoder_input")
    x = encoder_inputs
    
    for i, f in enumerate(FILTERS):
        x = layers.Conv1D(
            filters=f, 
            kernel_size=KERNEL_SIZES[i], 
            padding="same", 
            activation="tanh",
            kernel_regularizer=regularizers.l2(L2_REG)
        )(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(DROP_RATE)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dense(16, activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(x)
    
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def build_decoder(latent_dim, seq_len, num_channels):
    latent_inputs = layers.Input(shape=(latent_dim,), name="z_sampling")
    final_seq_len = seq_len // 8 
    
    x = layers.Dense(16, activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(latent_inputs)
    x = layers.Dense(64, activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dense(final_seq_len * FILTERS[-1], activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Reshape((final_seq_len, FILTERS[-1]))(x)
    
    for i in range(len(FILTERS)-1, -1, -1):
        x = layers.Conv1DTranspose(
            filters=FILTERS[i], 
            kernel_size=KERNEL_SIZES[i], 
            padding="same", 
            activation="tanh",
            kernel_regularizer=regularizers.l2(L2_REG)
        )(x)
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Dropout(DROP_RATE)(x)
        
    decoder_outputs = layers.Conv1D(
        filters=num_channels, 
        kernel_size=KERNEL_SIZES[0], 
        padding="same", 
        activation="linear", 
        name="reconstruction"
    )(x)
    
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

def build_classifier(latent_dim, num_classes):
    latent_inputs = layers.Input(shape=(latent_dim,), name="z_input")
    x = layers.Dense(64, activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(latent_inputs)
    x = layers.Dense(32, activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(x)
    outputs = layers.Dense(num_classes, activation="linear", name="class_output")(x)
    return keras.Model(latent_inputs, outputs, name="classifier")

class CVAE(keras.Model):
    def __init__(self, encoder, decoder, classifier, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.class_loss_tracker = keras.metrics.Mean(name="class_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker, self.class_loss_tracker]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            class_prediction = self.classifier(z)
            
            recon_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mse(x, reconstruction), axis=1))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            class_loss = tf.reduce_mean(keras.losses.mse(y, class_prediction))
            
            total_loss = recon_loss + (KL_WEIGHT * kl_loss) + (CLASS_WEIGHT * class_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(class_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
        }
    
    def test_step(self, data):
        x, y = data
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        class_prediction = self.classifier(z)
        
        recon_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mse(x, reconstruction), axis=1))
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        class_loss = tf.reduce_mean(keras.losses.mse(y, class_prediction))
        total_loss = recon_loss + (KL_WEIGHT * kl_loss) + (CLASS_WEIGHT * class_loss)
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "class_loss": class_loss,
        }

# ==========================================
# 3. ESECUZIONE MAIN
# ==========================================

if __name__ == "__main__":
    # A. Caricamento Dati
    print("Caricamento dati...")
    X, y = load_and_process_data(DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")

    # B. Costruzione Modello
    encoder = build_encoder(input_shape=(SEQ_LEN, len(CHANNELS)))
    decoder = build_decoder(LATENT_DIM, SEQ_LEN, len(CHANNELS))
    classifier = build_classifier(LATENT_DIM, NUM_CLASSES)
    
    cvae = CVAE(encoder, decoder, classifier)
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * len(X_train) // BATCH_SIZE
    )
    cvae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))

    # C. Training
    print("\nInizio addestramento CVAE...")
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    )
    
    history = cvae.fit(
        x=X_train, y=y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    # D. Salvataggio COMPLETO (Encoder, Decoder, Classifier)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    encoder_path = os.path.join(OUTPUT_DIR, "encoder_cvae.keras")
    decoder_path = os.path.join(OUTPUT_DIR, "decoder_cvae.keras")
    classifier_path = os.path.join(OUTPUT_DIR, "classifier_cvae.keras")

    encoder.save(encoder_path)
    decoder.save(decoder_path)
    classifier.save(classifier_path)

    print(f"\nModelli salvati in {OUTPUT_DIR}:")
    print(f"- Encoder: {encoder_path}")
    print(f"- Decoder: {decoder_path}")
    print(f"- Classifier: {classifier_path}")
    
    # E. Visualizzazione 3D
    print("Generazione grafico spazio latente 3D...")
    z_mean, _, _ = encoder.predict(X_val)
    labels_idx = np.argmax(y_val, axis=1)
    
    # Setup grafico 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Usiamo le prime 3 dimensioni dello spazio latente (0, 1, 2)
    scatter = ax.scatter(
        z_mean[:, 0], 
        z_mean[:, 1], 
        z_mean[:, 2], 
        c=labels_idx, 
        cmap='viridis', 
        alpha=0.7,
        s=20 # Dimensione punti
    )
    
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_zlabel("z[2]")
    ax.set_title("Spazio Latente CVAE (Prime 3 componenti)")
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Classe di Danno")
    
    plot_path = os.path.join(OUTPUT_DIR, "latent_space_preview_3d.png")
    plt.savefig(plot_path)
    print(f"Grafico 3D salvato in: {plot_path}")
    plt.show()
    
    print("\n--- Fase 1 Completata con successo! ---")