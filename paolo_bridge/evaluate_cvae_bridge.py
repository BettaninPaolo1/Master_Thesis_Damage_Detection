import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_absolute_error
from sklearn.decomposition import PCA

# ==========================================
# IMPOSTAZIONI GLOBALI GRAFICI (Font Ingranditi)
# ==========================================
plt.rcParams.update({
    'font.size': 16,          # Dimensione base
    'axes.titlesize': 20,     # Titolo del grafico
    'axes.labelsize': 20,     # Etichette assi (x, y, z)
    'xtick.labelsize': 14,    # Numeri asse x
    'ytick.labelsize': 14,    # Numeri asse y
    'legend.fontsize': 18,    # Testo legenda
    'figure.titlesize': 20    # Titolo figura globale
})

# ==========================================
# 0. CONFIGURAZIONE & CARTELLA OUTPUT
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models_evolution")

# --- CARTELLA RISULTATI ---
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_cvae_bridge")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_bridge.keras")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier_bridge.keras")

# IMPORTANTE: Usiamo la cartella di TEST
from data_loader_bridge import load_and_process_data, DATA_DIR_TEST

# ==========================================
# 1. CARICAMENTO MODELLI
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

def evaluate_bridge():
    print(f"--- Valutazione CVAE Bridge ---")
    print(f"I risultati verranno salvati in: {RESULTS_DIR}")
    
    # 1. Caricamento Dati
    X_test, y_test = load_and_process_data(DATA_DIR_TEST, is_training=False)
    
    # 2. Caricamento Modelli
    if not os.path.exists(ENCODER_PATH) or not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Modelli non trovati in {MODEL_DIR}.")
        
    keras.utils.get_custom_objects().update({'Sampling': Sampling})
    encoder = keras.models.load_model(ENCODER_PATH)
    classifier = keras.models.load_model(CLASSIFIER_PATH)
    
    # 3. Inferenza
    print("Esecuzione predizioni...")
    z_mean, _, _ = encoder.predict(X_test, verbose=0)
    y_pred = classifier.predict(z_mean, verbose=0)
    
    # ==========================================
    # 2. CALCOLO METRICHE
    # ==========================================
    
    true_class_idx = np.argmax(y_test, axis=1)
    pred_class_idx = np.argmax(y_pred, axis=1)
    true_intensity = np.max(y_test, axis=1)
    pred_intensity = np.max(y_pred, axis=1)
    
    acc = accuracy_score(true_class_idx, pred_class_idx)
    mae = mean_absolute_error(true_intensity, pred_intensity)
    r2 = r2_score(true_intensity, pred_intensity)
    
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("--- RISULTATI VALUTAZIONE BRIDGE ---\n")
        f.write(f"Accuracy Classificazione: {acc:.4f} ({acc:.2%})\n")
        f.write(f"MAE Intensità Danno:    {mae:.6f}\n")
        f.write(f"R2 Score Regressione:   {r2:.6f}\n")
    
    print(f"\n[METRICHE]")
    print(f"Accuracy: {acc:.2%}")
    print(f"MAE:      {mae:.4f}")
    print(f"Report salvato in: {metrics_path}")

    # ==========================================
    # 3. GRAFICI
    # ==========================================
    
    # --- A. Confusion Matrix ---
    cm = confusion_matrix(true_class_idx, pred_class_idx)
    plt.figure(figsize=(10, 8))
    # Aumentato size delle annotazioni interne
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                annot_kws={"size": 16},
                xticklabels=[f"D{i+1}" for i in range(6)], 
                yticklabels=[f"D{i+1}" for i in range(6)])
    plt.title(f"Confusion Matrix Bridge", pad=15)
    plt.xlabel("Predicted Class", labelpad=10)
    plt.ylabel("True Class", labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    
    # --- B. Regression Plot ---
    plt.figure(figsize=(10, 8))
    
    plt.scatter(true_intensity, pred_intensity, 
                alpha=0.6, c='dodgerblue', edgecolor='k', s=80, label='Predicted vs True')
    
    plt.plot([0, 0.6], [0, 0.6], 'k--', lw=2, label='Ideal')
    
    plt.title(f"Regression Damage Intensity", pad=15)
    plt.xlabel("True Damage Level", labelpad=10)
    plt.ylabel("Predicted Damage Level", labelpad=10)
    plt.xlim(0, 0.6)
    plt.ylim(0, 0.6)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "regression_plot.png"))
    plt.close()

    # --- C. 3D LATENT SPACE PLOT ---
    print("Generazione Plot 3D...")
    
    if z_mean.shape[1] > 3:
        pca = PCA(n_components=3)
        z_3d = pca.fit_transform(z_mean)
        print(f"PCA applicata: {z_mean.shape[1]}D -> 3D")
    else:
        z_3d = z_mean[:, :3]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2],
                         c=true_class_idx, 
                         cmap='viridis',      
                         s=40, # Punti leggermente più grandi
                         alpha=0.8, 
                         edgecolors='none') 
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, aspect=20, pad=0.05)
    cbar.set_label('Classes', rotation=90, labelpad=15, fontsize=16)
    
    unique_classes = np.unique(true_class_idx)
    cbar.set_ticks(unique_classes)
    cbar.set_ticklabels(unique_classes.astype(int), fontsize=14)

    ax.set_title("3D Latent Space - TUTTE LE CLASSI", pad=20)
    
    # Aumentato il padding per distanziare le label dall'asse 3D
    ax.set_xlabel("PCA[0]", labelpad=15)
    ax.set_ylabel("PCA[1]", labelpad=15)
    ax.set_zlabel("PCA[2]", labelpad=15)
    
    # Aumentati i tick numbers per i 3 assi
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

    ax.view_init(elev=30, azim=-60)
    
    # Doppio salvataggio come avevi richiesto
    latent_path_replica = os.path.join(RESULTS_DIR, "latent_space_3d_replica.png")
    plt.savefig(latent_path_replica, dpi=150)
    plt.show()
    
    latent_path = os.path.join(RESULTS_DIR, "latent_space_3d.png")
    plt.savefig(latent_path, dpi=150)
    plt.close()
    
    print(f"Grafici salvati in: {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate_bridge()