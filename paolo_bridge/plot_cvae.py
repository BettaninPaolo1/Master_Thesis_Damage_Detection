import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.decomposition import PCA

# ==========================================
# CONFIGURAZIONE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models_evolution")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_latent_analysis")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_bridge.keras")
from data_loader_bridge import load_and_process_data, DATA_DIR_TEST

# ==========================================
# DEFINIZIONE SAMPLING LAYER
# ==========================================
try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ==========================================
# FUNZIONE DI PLOT
# ==========================================
def plot_latent_space_pca():
    print("--- Analisi Spazio Latente (PCA) ---")
    
    # 1. Caricamento Dati
    print("Caricamento dati di test...")
    X_test, y_test = load_and_process_data(DATA_DIR_TEST, is_training=False)
    
    # Ricaviamo le etichette di classe (0, 1, ..., 5)
    labels = np.argmax(y_test, axis=1)
    class_names = [f"D{i+1}" for i in np.unique(labels)]
    
    # 2. Caricamento Encoder
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder non trovato in: {ENCODER_PATH}")
        
    print("Caricamento Encoder...")
    keras.utils.get_custom_objects().update({'Sampling': Sampling})
    encoder = keras.models.load_model(ENCODER_PATH)
    
    # 3. Proiezione nello spazio latente
    print("Calcolo vettori latenti (z_mean)...")
    z_mean, _, _ = encoder.predict(X_test, verbose=0)
    
    print(f"Dimensione originale spazio latente: {z_mean.shape[1]}")

    # ==========================================
    # PLOT PCA 2D (Migliore per report stampati)
    # ==========================================
    print("Generazione PCA 2D...")
    pca_2d = PCA(n_components=2)
    z_2d = pca_2d.fit_transform(z_mean)
    var_2d = np.sum(pca_2d.explained_variance_ratio_)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k', s=40)
    plt.colorbar(scatter, ticks=np.unique(labels), label='Damage Class', format='D%d')
    
    plt.title(f"Latent Space PCA 2D (Expl. Var: {var_2d:.2%})", fontsize=14)
    plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})")
    plt.grid(True, alpha=0.3)
    
    save_path_2d = os.path.join(RESULTS_DIR, "latent_space_pca_2d.png")
    plt.savefig(save_path_2d, dpi=300)
    plt.close()
    print(f"Salvato: {save_path_2d}")

    # ==========================================
    # PLOT PCA 3D (Migliore per visualizzazione interattiva)
    # ==========================================
    print("Generazione PCA 3D...")
    pca_3d = PCA(n_components=3)
    z_3d = pca_3d.fit_transform(z_mean)
    var_3d = np.sum(pca_3d.explained_variance_ratio_)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], 
                         c=labels, cmap='viridis', s=30, alpha=0.8, edgecolors='k', linewidth=0.2)
    
    # Legenda colori
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Damage Class')
    cbar.set_ticks(np.unique(labels))
    cbar.set_ticklabels(class_names)

    ax.set_title(f"Latent Space PCA 3D (Expl. Var: {var_3d:.2%})", fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    
    # Imposta un angolo di vista iniziale (opzionale)
    ax.view_init(elev=30, azim=135)

    save_path_3d = os.path.join(RESULTS_DIR, "latent_space_pca_3d.png")
    plt.savefig(save_path_3d, dpi=300)
    plt.close()
    print(f"Salvato: {save_path_3d}")
    
    print("\nFinito. Controlla la cartella 'results_latent_analysis'.")

if __name__ == "__main__":
    plot_latent_space_pca()