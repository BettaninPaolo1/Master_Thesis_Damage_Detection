import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CONFIGURAZIONE (TARGET: SVGD 8D)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cartella di output (deve corrispondere a quella di run_svgd_8d.py)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_svgd_8d_deepfilm") 

# Nome file CSV
CSV_FILENAME = "results_svgd_8d_deepfilm.csv"
CSV_PATH = os.path.join(RESULTS_DIR, CSV_FILENAME)

# Numero classi
NUM_CLASSES = 7

# ==========================================
# 2. UTILS DI PARSING
# ==========================================
def parse_vector_col(df, col_name):
    """
    Converte le stringhe del CSV (es: '[0.1 0.2 ...]') in array numpy.
    """
    try:
        # Rimuove parentesi quadre, newline e virgole per il parsing sicuro
        return df[col_name].apply(lambda x: np.fromstring(
            str(x).replace('[','').replace(']','').replace('\n',' ').replace(',', ' '), 
            sep=' '
        ))
    except Exception as e:
        print(f"Errore parsing colonna {col_name}: {e}")
        return None

def get_metrics(y_true, y_pred):
    """Calcola metriche standard per vettori."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "CosSim": np.mean(cosine_similarity(y_true, y_pred).diagonal())
    }

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Imposta stile grafico
    sns.set_theme(style="whitegrid")

    plt.rcParams.update({
    'font.size': 16,          # Dimensione base
    'axes.titlesize': 20,     # Titolo del grafico
    'axes.labelsize': 18,     # Etichette assi (x, y, z)
    'xtick.labelsize': 14,    # Numeri asse x
    'ytick.labelsize': 14,    # Numeri asse y
    'legend.fontsize': 18,    # Testo legenda
    'figure.titlesize': 20    # Titolo figura globale
})
    
    # Check esistenza file
    if not os.path.exists(CSV_PATH):
        print(f"\n[ERRORE] Il file dati non esiste: {CSV_PATH}")
        print("Esegui prima 'run_svgd_8d.py' per generare i risultati.\n")
        exit()

    print(f"--- Lettura Dati da: {CSV_PATH} ---")
    df = pd.read_csv(CSV_PATH)
    
    # ------------------------------------------
    # A. PARSING VETTORI
    # ------------------------------------------
    print("Parsing colonne vettoriali in corso...")
    y_true = np.stack(parse_vector_col(df, 'True_Vector').values)
    y_mean = np.stack(parse_vector_col(df, 'Mean_Vector').values)
    y_mode = np.stack(parse_vector_col(df, 'Mode_Vector').values)
    y_std  = np.stack(parse_vector_col(df, 'Std_Vector').values) 
    y_vae  = np.stack(parse_vector_col(df, 'VAE_Init').values)

    # ------------------------------------------
    # B. PREPARAZIONE DATI
    # ------------------------------------------
    # Indici delle classi (0-6)
    idx_range = np.arange(len(y_true))
    true_cls_idx = np.argmax(y_true, axis=1)
    mean_cls_idx = np.argmax(y_mean, axis=1)
    mode_cls_idx = np.argmax(y_mode, axis=1)

    # Label Classi (1-7 per display)
    true_cls = true_cls_idx + 1
    mean_cls = mean_cls_idx + 1
    mode_cls = mode_cls_idx + 1
    
    # --- LIVELLI DI DANNO ---
    true_levels = y_true[idx_range, true_cls_idx]
    vae_levels = y_vae[idx_range, true_cls_idx]
    

    # Predicted Levels (Scalar based on Predicted Class)
    pred_levels_mean_selected = y_mean[idx_range, mean_cls_idx]
    pred_levels_mode_selected = y_mode[idx_range, mode_cls_idx]

    # CVAE Predicted Class and Levels
    vae_cls_idx = np.argmax(y_vae, axis=1)
    pred_levels_cvae_selected = y_vae[idx_range, vae_cls_idx]

    # NUOVO: MAE scalare per il CVAE
    mae_scalar_level_cvae = mean_absolute_error(true_levels, pred_levels_cvae_selected)

    # Calcolo MAE scalare (solo sul livello di danno nella classe predetta vs classe reale)
    mae_scalar_level_mean = mean_absolute_error(true_levels, pred_levels_mean_selected)
    mae_scalar_level_mode = mean_absolute_error(true_levels, pred_levels_mode_selected)

    

    # Metriche per singolo campione (per scatter plot)
    mae_sample_mean = np.mean(np.abs(y_true - y_mean), axis=1)
    mae_sample_mode = np.mean(np.abs(y_true - y_mode), axis=1)

    r2_reg_mean = r2_score(true_levels, pred_levels_mean_selected)
    r2_reg_mode = r2_score(true_levels, pred_levels_mode_selected)
    
    # --- INCERTEZZA ---
    # 1. Incertezza MEDIA (su tutte le classi) -> Per Plot 6, 7, 8
    avg_uncertainty = np.mean(y_std, axis=1)
    
    # 2. Incertezza SPECIFICA (solo della classe predetta dalla Moda) -> Per Plot 9, 10
    pred_uncertainty = y_std[idx_range, mode_cls_idx]

    # Maschere per predizioni corrette/errate (Moda)
    mask_ok = (true_cls == mode_cls)

    # ------------------------------------------
    # C. REPORT TESTUALE
    # ------------------------------------------
    met_mean = get_metrics(y_true, y_mean)
    met_mode = get_metrics(y_true, y_mode)
    acc_mode = accuracy_score(true_cls, mode_cls)
    acc_mean = accuracy_score(true_cls, mean_cls)

    print("\n" + "="*65)
    print(f"{'METRICA':<10} | {'MEAN Est':<12} | {'MODE Est':<12} | {'WINNER'}")
    print("-" * 65)
    print(f"{'MAE':<10} | {met_mean['MAE']:.5f}      | {met_mode['MAE']:.5f}      | {'<-- Mode' if met_mode['MAE'] < met_mean['MAE'] else 'Mean'}")
    print(f"{'R2':<10}  | {met_mean['R2']:.5f}      | {met_mode['R2']:.5f}      | {'<-- Mode' if met_mode['R2'] > met_mean['R2'] else 'Mean'}")
    print(f"{'CosSim':<10} | {met_mean['CosSim']:.5f}      | {met_mode['CosSim']:.5f}      | {'<-- Mode' if met_mode['CosSim'] > met_mean['CosSim'] else 'Mean'}")
    print("-" * 65)
    print(f"Accuratezza Classificazione (Mean): {acc_mean:.2%}")
    print(f"Accuratezza Classificazione (Mode): {acc_mode:.2%}")

    # ==========================================
    # D. GENERAZIONE GRAFICI (10 PLOT)
    # ==========================================

    # --- PLOT 1: Confusion Matrix (MEDIA) ---
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(true_cls, mean_cls, labels=range(1, NUM_CLASSES + 1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=range(1, NUM_CLASSES + 1), 
                yticklabels=range(1, NUM_CLASSES + 1))
    plt.title(f'Confusion Matrix Beam - SVGD + FiLM')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_1_confusion_matrix_MEAN.png"))
    plt.close()

    # --- PLOT 2: Regression (MODA) ---
    plt.figure(figsize=(9, 8))
    plt.scatter(true_levels[mask_ok], pred_levels_mode_selected[mask_ok], 
                alpha=0.6, c='dodgerblue', edgecolor='k', s=70, label='Correct Class')
    plt.scatter(true_levels[~mask_ok], pred_levels_mode_selected[~mask_ok], 
                alpha=0.8, c='crimson', marker='X', s=90, label='Wrong Class')
    plt.plot([0, 0.6], [0, 0.6], 'k--', lw=2, label='Ideal')
    plt.title(f'Regression: True Level vs Selected Class Level')
    plt.xlabel('True Damage Level'); plt.ylabel('Predicted Level (of Selected Class)')
    plt.legend(); plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_2_regression_mode.png"))
    plt.close()

    # --- PLOT 3: Regression (MEDIA) ---
    plt.figure(figsize=(9, 8))
    mask_ok_mean = (true_cls == mean_cls)
    plt.scatter(true_levels[mask_ok_mean], pred_levels_mean_selected[mask_ok_mean], 
                alpha=0.6, c='mediumseagreen', edgecolor='k', s=70, label='Correct Class')
    plt.scatter(true_levels[~mask_ok_mean], pred_levels_mean_selected[~mask_ok_mean], 
                alpha=0.8, c='crimson', marker='X', s=90, label='Wrong Class')
    plt.plot([0, 0.6], [0, 0.6], 'k--', lw=2, label='Ideal')
    plt.title(f'Regression: True Level vs Predicted Level')
    plt.xlabel('True Damage Level'); plt.ylabel('Predicted Damage Level ')
    plt.legend(); plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_3_regression_mean.png"))
    plt.close()

    # --- PLOT 4: Confronto Errori (Head-to-Head) ---
    plt.figure(figsize=(9, 8))
    plt.scatter(mae_sample_mean, mae_sample_mode, alpha=0.6, c='purple', edgecolor='k', s=60)
    max_err = max(mae_sample_mean.max(), mae_sample_mode.max()) * 1.05
    plt.plot([0, max_err], [0, max_err], 'r--', lw=2, label='Parity Line')
    plt.fill_between([0, max_err], [0, max_err], 0, color='green', alpha=0.1, label='Mode is Better')
    plt.fill_between([0, max_err], [0, max_err], max_err, color='orange', alpha=0.1, label='Mean is Better')
    plt.xlim(0, max_err); plt.ylim(0, max_err)
    plt.xlabel('MAE (MEAN)'); plt.ylabel('MAE (MODE)')
    plt.title('Head-to-Head Error Comparison')
    plt.legend(loc='upper left'); plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_4_error_comparison.png"))
    plt.close()

    # --- PLOT 5: VAE vs SVGD Shift (MEAN) ---
    plt.figure(figsize=(10, 8))
    plt.scatter(true_levels, vae_levels, marker='x', c='gray', s=50, label='VAE Start', alpha=0.5)
    plt.scatter(true_levels, pred_levels_mean_selected, marker='o', c='mediumseagreen', s=60, 
                label='SVGD End (Mean)', edgecolor='k')
    for i in range(len(true_levels)):
        if abs(pred_levels_mean_selected[i] - vae_levels[i]) > 0.015:
            plt.plot([true_levels[i], true_levels[i]], 
                     [vae_levels[i], pred_levels_mean_selected[i]], 
                     c='gray', alpha=0.2)
    plt.plot([0, 0.6], [0, 0.6], 'k--', lw=2)
    plt.title('Refinement Impact: VAE vs SVGD (MEAN)')
    plt.xlabel('True Damage Level')
    plt.ylabel('Estimated Level (Selected Class - Mean)')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_5_vae_shift_MEAN.png"))
    plt.close()

    # --------------------------------------------------------------------------------
    # PLOT GRUPPO A: INCERTEZZA MEDIA (Overall Confusion)
    # --------------------------------------------------------------------------------

    # --- PLOT 6: Scatter Errore vs Incertezza (MEDIA) ---
    plt.figure(figsize=(10, 8))
    if len(avg_uncertainty) > 1:
        corr, _ = pearsonr(avg_uncertainty, mae_sample_mode)
    else: corr = 0
    
    plt.scatter(avg_uncertainty[mask_ok], mae_sample_mode[mask_ok], 
                c='mediumseagreen', alpha=0.6, label='Correct Class', edgecolor='k', s=60)
    plt.scatter(avg_uncertainty[~mask_ok], mae_sample_mode[~mask_ok], 
                c='crimson', alpha=0.8, label='Wrong Class', marker='X', edgecolor='k', s=90)
    
    if len(avg_uncertainty) > 1:
        z = np.polyfit(avg_uncertainty, mae_sample_mode, 1)
        p = np.poly1d(z)
        plt.plot(avg_uncertainty, p(avg_uncertainty), "k--", alpha=0.6, lw=2, label=f'Trend (Corr: {corr:.2f})')

    plt.title('Calibration: Error vs AVERAGE Uncertainty', fontsize=14)
    plt.xlabel('Average Uncertainty (All Classes)', fontsize=12)
    plt.ylabel('Actual Error (MAE Mode)', fontsize=12)
    plt.legend(loc='upper left'); plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_6_uncertainty_MEAN_vs_error.png"))
    plt.close()

    # --- PLOT 7: Boxplot Incertezza (MEDIA) ---
    plt.figure(figsize=(8, 6))
    data_to_plot = [avg_uncertainty[mask_ok], avg_uncertainty[~mask_ok]]
    if len(avg_uncertainty[~mask_ok]) > 0:
        bplot = plt.boxplot(data_to_plot, patch_artist=True, labels=['Correct Predictions', 'Wrong Predictions'])
        colors = ['mediumseagreen', 'crimson']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
    else: plt.text(0.5, 0.5, "No Wrong Predictions!", ha='center', fontsize=14)
    
    plt.title('Correct vs Wrong: AVERAGE Uncertainty Distribution', fontsize=14)
    plt.ylabel('Average Uncertainty', fontsize=12)
    plt.grid(True, axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_7_uncertainty_MEAN_boxplot.png"))
    plt.close()

    # --- PLOT 8: Incertezza Media per Classe Predetta ---
    plt.figure(figsize=(10, 6))
    avg_unc_per_class = []
    class_labels = range(1, NUM_CLASSES + 1)
    counts = []
    for c in class_labels:
        mask_c = (mode_cls == c)
        if np.sum(mask_c) > 0:
            mean_val = np.mean(avg_uncertainty[mask_c])
            count = np.sum(mask_c)
        else: mean_val = 0; count = 0
        avg_unc_per_class.append(mean_val)
        counts.append(count)

    bars = plt.bar(class_labels, avg_unc_per_class, color=sns.color_palette("viridis", NUM_CLASSES), alpha=0.8, edgecolor='k')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}\n(n={count})',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.title('Average Uncertainty per Predicted Class', fontsize=14)
    plt.xlabel('Predicted Class'); plt.ylabel('Avg Uncertainty')
    plt.xticks(class_labels); plt.grid(True, axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_8_uncertainty_per_class.png"))
    plt.close()

    # --------------------------------------------------------------------------------
    # PLOT GRUPPO B: INCERTEZZA PREDICTED CLASS (Specific Confusion)
    # --------------------------------------------------------------------------------

    # --- PLOT 9: Scatter Errore vs Incertezza (PREDICTED CLASS) ---
    plt.figure(figsize=(10, 8))
    if len(pred_uncertainty) > 1:
        corr_pred, _ = pearsonr(pred_uncertainty, mae_sample_mode)
    else: corr_pred = 0
    
    plt.scatter(pred_uncertainty[mask_ok], mae_sample_mode[mask_ok], 
                c='dodgerblue', alpha=0.6, label='Correct Class', edgecolor='k', s=60)
    plt.scatter(pred_uncertainty[~mask_ok], mae_sample_mode[~mask_ok], 
                c='orange', alpha=0.8, label='Wrong Class', marker='X', edgecolor='k', s=90)
    
    if len(pred_uncertainty) > 1:
        z = np.polyfit(pred_uncertainty, mae_sample_mode, 1)
        p = np.poly1d(z)
        plt.plot(pred_uncertainty, p(pred_uncertainty), "k--", alpha=0.6, lw=2, label=f'Trend (Corr: {corr_pred:.2f})')

    plt.title('Calibration: Error vs PREDICTED CLASS Uncertainty', fontsize=14)
    plt.xlabel('Uncertainty of Predicted Class (Std Dev)', fontsize=12)
    plt.ylabel('Actual Error (MAE Mode)', fontsize=12)
    plt.legend(loc='upper left'); plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_9_uncertainty_PRED_vs_error.png"))
    plt.close()

    # --- PLOT 10: Boxplot Incertezza (PREDICTED CLASS) ---
    plt.figure(figsize=(8, 6))
    data_to_plot_pred = [pred_uncertainty[mask_ok], pred_uncertainty[~mask_ok]]
    
    if len(pred_uncertainty[~mask_ok]) > 0:
        bplot = plt.boxplot(data_to_plot_pred, patch_artist=True, labels=['Correct Predictions', 'Wrong Predictions'])
        colors = ['dodgerblue', 'orange']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
    else: plt.text(0.5, 0.5, "No Wrong Predictions!", ha='center', fontsize=14)
    
    plt.title('Correct vs Wrong: PREDICTED CLASS Uncertainty', fontsize=14)
    plt.ylabel('Uncertainty of Predicted Class', fontsize=12)
    plt.grid(True, axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_10_uncertainty_PRED_boxplot.png"))
    plt.close()

    print(f"\n[DONE] Generati 10 grafici in:\n{RESULTS_DIR}")

    # =========================================================================
    # NUOVO BLOCCO: CONFRONTO DIRETTO CVAE vs SVGD (MODE)
    # =========================================================================
    
    # 1. Calcoliamo la classe predetta dal CVAE (argmax del vettore iniziale)
    vae_cls_idx = np.argmax(y_vae, axis=1)
    vae_cls = vae_cls_idx + 1  # Portiamo a range 1-7
    
    # 2. Calcolo dei casi specifici (Usiamo SVGD MODE come riferimento finale)
    # Caso A: Il CVAE aveva sbagliato, ma SVGD ha corretto il tiro (Miglioramento)
    improved_mask = (vae_cls != true_cls) & (mode_cls == true_cls)
    n_improved = np.sum(improved_mask)

    # Caso B: Il CVAE aveva indovinato, ma SVGD ha sbagliato (Peggioramento)
    degraded_mask = (vae_cls == true_cls) & (mode_cls != true_cls)
    n_degraded = np.sum(degraded_mask)

    # Caso C: Entrambi corretti
    both_correct = np.sum((vae_cls == true_cls) & (mode_cls == true_cls))

    # Caso D: Entrambi sbagliati
    both_wrong = np.sum((vae_cls != true_cls) & (mode_cls != true_cls))

    print("\n" + "="*65)
    print(f"DUELLO: CVAE (Initial) vs SVGD (Refined Mode)")
    print("-" * 65)
    print(f"1. RECUPERATI (CVAE Sbagliato -> SVGD Giusto): {n_improved} campioni (Keep!)")
    print(f"2. PERSI      (CVAE Giusto    -> SVGD Sbagliato): {n_degraded} campioni (Warning)")
    print("-" * 65)
    print(f"Bilancio Netto (Recuperati - Persi): {n_improved - n_degraded:+d}")
    print(f"Entrambi Corretti: {both_correct}")
    print(f"Entrambi Sbagliati: {both_wrong}")
    print("="*65)

    # -------------------------------------------------------------------------
    # PLOT 11: MAE per Classe Reale (SOLO MEAN ESTIMATOR)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    mae_per_class_mean = []
    unc_per_class_mean = [] # Incertezza media per classe
    
    class_labels = range(1, NUM_CLASSES + 1)
    
    for c in class_labels:
        mask_c = (true_cls == c)
        if np.sum(mask_c) > 0:
            mae_per_class_mean.append(np.mean(mae_sample_mean[mask_c]))
            unc_per_class_mean.append(np.mean(y_std[mask_c]))
        else:
            mae_per_class_mean.append(0.0)
            unc_per_class_mean.append(0.0)

    # Plot solo barre Mean
    plt.bar(class_labels, mae_per_class_mean, color='mediumseagreen', alpha=0.8, edgecolor='k', width=0.6)
    
    plt.title('Mean Absolute Error (MAE) per True Class (Mean Estimator)', fontsize=14)
    plt.xlabel('True Class Index')
    plt.ylabel('Average MAE')
    plt.xticks(class_labels)
    plt.grid(True, axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_11_mae_per_class_MEAN_ONLY.png"))
    plt.close()

    # =========================================================================
    # E. CALCOLO METRICHE AGGIORNATE E SCRITTURA FILE TESTUALE
    # =========================================================================
    SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary_statistics.txt")
    
    # 1. Calcoli preliminari Generali
    global_avg_uncertainty = np.mean(y_std) # Media globale di tutte le deviazioni standard
    avg_time = df['Time_Seconds'].mean() if 'Time_Seconds' in df.columns else df['Time'].mean()

    # 2. CALCOLO R2 "REGRESSION STYLE" (Scalar vs Scalar)
    # Usiamo lo stesso approccio dei plot 2 e 3: True Level vs Level della classe predetta
    # true_levels è già stato calcolato: y_true[idx, true_cls_idx]
    # pred_levels_mean_selected è già calcolato: y_mean[idx, mean_cls_idx]
    # pred_levels_mode_selected è già calcolato: y_mode[idx, mode_cls_idx]
    
    

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("===================================================================\n")
        f.write("             RIEPILOGO STATISTICO COMPLETO SVGD 8D\n")
        f.write("===================================================================\n\n")
        
        f.write(f"Timestamp Analisi: {pd.Timestamp.now()}\n")
        f.write(f"Numero totale campioni: {len(df)}\n")
        f.write(f"Tempo medio esecuzione (per sample): {avg_time:.4f} s\n")
        f.write(f"Incertezza Media Globale (Avg Std_Vector): {global_avg_uncertainty:.6f}\n\n")
        
        f.write("-" * 85 + "\n")
        f.write(f"{'METRICA':<15} | {'MEAN Estimator (Avg)':<25} | {'MODE Estimator (Avg)':<25}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'MAE (Vector)':<15} | {met_mean['MAE']:.6f}                  | {met_mode['MAE']:.6f}\n")
        f.write(f"{'MAE (Intensity)':<15} | {mae_scalar_level_mean:.6f}                  | {mae_scalar_level_mode:.6f}\n")
        f.write(f"{'CVAE MAE (Intensity)':<15} | {mae_scalar_level_cvae:.6f}                  | -- \n")
        f.write(f"{'MSE':<15} | {df['MSE_Mean'].mean():.6f}                  | {df['MSE_Mode'].mean():.6f}\n")
        f.write(f"{'RMSE':<15} | {np.sqrt(df['MSE_Mean']).mean():.6f}                  | {np.sqrt(df['MSE_Mode']).mean():.6f}\n")
        
        # Qui usiamo le nuove variabili r2_reg_...
        f.write(f"{'R2 (Regress.)':<15} | {r2_reg_mean:.6f}                  | {r2_reg_mode:.6f}\n")
        
        f.write(f"{'Cos Sim':<15} | {df['CosSim_Mean'].mean():.6f}                  | {df['CosSim_Mode'].mean():.6f}\n")
        f.write(f"{'SMAPE':<15} | {df['SMAPE_Mean'].mean():.6f}                  | {df['SMAPE_Mode'].mean():.6f}\n")
        f.write("-" * 85 + "\n\n")
        
        f.write("--- PERFORMANCE CLASSIFICAZIONE ---\n")
        f.write(f"Accuratezza (Mean): {acc_mean:.2%} ({int(acc_mean*len(df))}/{len(df)})\n")
        f.write(f"Accuratezza (Mode): {acc_mode:.2%} ({int(acc_mode*len(df))}/{len(df)})\n\n")
        
        f.write("--- DETTAGLIO PER CLASSE REALE (Mean Estimator) ---\n")
        f.write(f"{'Class':<6} | {'Avg MAE':<12} | {'Avg Uncertainty (Std)':<22} | {'Count':<6}\n")
        f.write("-" * 60 + "\n")
        for i, c in enumerate(class_labels):
            count_c = np.sum(true_cls == c)
            f.write(f"{c:<6} | {mae_per_class_mean[i]:.6f}      | {unc_per_class_mean[i]:.6f}                 | {count_c:<6}\n")
        f.write("\n")

        f.write("--- ANALISI REFINEMENT (CVAE vs SVGD MODE) ---\n")
        f.write(f"Migliorati (CVAE err -> SVGD ok): {n_improved}\n")
        f.write(f"Peggiorati (CVAE ok  -> SVGD err): {n_degraded}\n")
        f.write(f"Bilancio Netto: {n_improved - n_degraded:+d}\n\n")

        f.write("--- ELENCO CAMPIONI CON CLASSIFICAZIONE ERRATA (MODE) ---\n")
        wrong_df = df[df['True_Class'] != df['Pred_Class_Mode']].copy()
        
        if len(wrong_df) > 0:
            f.write(f"Totale Errori: {len(wrong_df)}\n")
            # Recuperiamo l'incertezza specifica della classe predetta per questi campioni
            mask_wrong = (true_cls != mode_cls)
            unc_wrong = pred_uncertainty[mask_wrong]
            
            f.write(f"{'Index':<8} | {'True':<6} | {'Pred':<6} | {'MAE (Mode)':<12} | {'Uncertainty (Pred Class)':<24}\n")
            f.write("-" * 75 + "\n")
            
            # Iteriamo sugli array filtrati
            idxs = df.index[mask_wrong]
            t_cls = true_cls[mask_wrong]
            p_cls = mode_cls[mask_wrong]
            maes = mae_sample_mode[mask_wrong]
            
            for idx_val, t, p, m, u in zip(idxs, t_cls, p_cls, maes, unc_wrong):
                 f.write(f"{df.iloc[idx_val]['Index']:<8} | {t:<6} | {p:<6} | {m:.6f}       | {u:.6f}\n")
        else:
            f.write("Nessun errore di classificazione riscontrato.\n")

    print(f"Statistiche complete salvate in: {SUMMARY_PATH}")