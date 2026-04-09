import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, ops
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==========================================
# 0. CONFIGURAZIONE SEED & PERCORSI
# ==========================================
SEED = 964051881 # Usiamo il tuo seed "vincitore"
SEED = random.randint(0, 2**32 - 1)

print(f"\n{'='*40}")
print(f"   ESECUZIONE CVAE - LATENT DIM: 8")
print(f"   SEED: {SEED}")
print(f"{'='*40}\n")

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models_evolution_v2") # Nuova cartella per v2

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

from data_loader import load_and_process_data, DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR

# ==========================================
# 1. IPERPARAMETRI AGGIORNATI (8 LAYERS)
# ==========================================
LATENT_DIM = 8          # <--- Aumentato per maggiore separabilità
NUM_CLASSES = 7         
BATCH_SIZE = 32
EPOCHS = 300            # Leggermente più lungo per stabilizzare z
LEARNING_RATE = 1e-3
KL_WEIGHT = 1.2         # Leggera enfasi in più sulla regolarizzazione latente
CLASS_WEIGHT = 4000.0   # Aumentata per forzare la separazione delle classi

FILTERS = [32, 64, 128] # Più filtri nell'ultimo layer per catturare feature fini
KERNEL_SIZES = [25, 13, 7]
DROP_RATE = 0.08
L2_REG = 5e-4           # Ridotta per non "soffocare" troppo l'encoder

# ==========================================
# 2. DEFINIZIONE DEL MODELLO
# ==========================================

class Sampling(layers.Layer):
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
            filters=f, kernel_size=KERNEL_SIZES[i], 
            padding="same", activation="tanh",
            kernel_regularizer=regularizers.l2(L2_REG)
        )(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(DROP_RATE)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dense(64, activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(x)
    
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder_8d")

def build_decoder(latent_dim, seq_len, num_channels):
    latent_inputs = layers.Input(shape=(latent_dim,), name="z_sampling")
    final_seq_len = seq_len // 8 
    
    x = layers.Dense(64, activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(latent_inputs)
    x = layers.Dense(final_seq_len * FILTERS[-1], activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Reshape((final_seq_len, FILTERS[-1]))(x)
    
    for i in range(len(FILTERS)-1, -1, -1):
        x = layers.Conv1DTranspose(
            filters=FILTERS[i], kernel_size=KERNEL_SIZES[i], 
            padding="same", activation="tanh",
            kernel_regularizer=regularizers.l2(L2_REG)
        )(x)
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Dropout(DROP_RATE)(x)
        
    decoder_outputs = layers.Conv1D(
        filters=num_channels, kernel_size=KERNEL_SIZES[0], 
        padding="same", activation="linear", name="reconstruction"
    )(x)
    
    return keras.Model(latent_inputs, decoder_outputs, name="decoder_8d")

def build_classifier(latent_dim, num_classes):
    latent_inputs = layers.Input(shape=(latent_dim,), name="z_input")
    x = layers.Dense(64, activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(latent_inputs)
    x = layers.Dense(32, activation="tanh", kernel_regularizer=regularizers.l2(L2_REG))(x)
    outputs = layers.Dense(num_classes, activation="linear", name="class_output")(x)
    return keras.Model(latent_inputs, outputs, name="classifier_8d")

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
        return {m.name: m.result() for m in self.metrics}
    
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
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss, "class_loss": class_loss}

# ==========================================
# 3. MAIN
# ==========================================

if __name__ == "__main__":
    X, y = load_and_process_data(DATA_DIR, CHANNELS, SEQ_LEN, ADDED_SNR)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    encoder = build_encoder(input_shape=(SEQ_LEN, len(CHANNELS)))
    decoder = build_decoder(LATENT_DIM, SEQ_LEN, len(CHANNELS))
    classifier = build_classifier(LATENT_DIM, NUM_CLASSES)
    
    cvae = CVAE(encoder, decoder, classifier)
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * len(X_train) // BATCH_SIZE
    )
    cvae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    
    history = cvae.fit(x=X_train, y=y_train, validation_data=(X_val, y_val),
                       epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop])

    # Salvataggio con nomi file aggiornati (v2 / 8D)
    encoder_path = os.path.join(OUTPUT_DIR, "encoder_cvae_8d_old.keras")
    decoder_path = os.path.join(OUTPUT_DIR, "decoder_cvae_8d_old.keras")
    classifier_path = os.path.join(OUTPUT_DIR, "classifier_cvae_8d_old.keras")

    encoder.save(encoder_path)
    decoder.save(decoder_path)
    classifier.save(classifier_path)

    # Plot 3D Spazio Latente
    z_mean, _, _ = encoder.predict(X_val)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], c=np.argmax(y_val, axis=1), cmap='viridis')
    plt.title("Visualizzazione 3D dello Spazio Latente 8D (Prime 3 Dim)")
    plt.savefig(os.path.join(OUTPUT_DIR, "latent_space_8d_preview.png"))
    plt.show()

    print(f"\n--- Training 8D Completato! Modelli in {OUTPUT_DIR} ---")