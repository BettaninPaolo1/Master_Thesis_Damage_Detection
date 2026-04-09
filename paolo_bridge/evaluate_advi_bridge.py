import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cartella di output (deve corrispondere a quella dello script ADVI)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_advi_bridge_stats") 

# Nome file CSV
CSV_FILENAME = "results_batch_data_advi.csv"
CSV_PATH = os.path.join(RESULTS_DIR, CSV_FILENAME)

# Numero classi
NUM_CLASSES = 6

# ==========================================
# 2. UTILS DI PARSING E METRICHE
# ==========================================
def parse_vector_col(df, col_name):
    """
    Converte le stringhe del CSV in array numpy.
    Gestisce formattazione con newlines e bracket.
    """
    try:
        return df[col_name].apply(lambda x: np.fromstring(
            str(x).replace('[','').replace(']','').replace('\n',' ').replace(',', ' '), 
            sep=' '
        ))
    except Exception as e:
        print(f"Errore parsing colonna {col_name}: {e}")
        return None

def get_metrics(y_true, y_pred):
    """Calcola metriche standard vettoriali."""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    cos_sim_vals = np.diag(cosine_similarity(y_true, y_pred))
    cos_sim = np.mean(cos_sim_vals)
    return {"MAE": mae, "R2": r2, "CosSim": cos_sim}

def smape(a, b): 
    """Calcola SMAPE gestendo la divisione per zero."""
    return np.mean(np.abs(a-b) / ((np.abs(a)+np.abs(b))/2 + 1e-10)) * 100

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    
    if not os.path.exists(CSV_PATH):
        print(f"\n[ERRORE] Il file dati non esiste: {CSV_PATH}")
        print("Esegui prima lo script ADVI per generare i risultati.\n")
        exit()

    print(f"--- Lettura Dati da: {CSV_FILENAME} ---")
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
    idx_range = np.arange(len(y_true))
    
    # Indici Classi
    true_cls_idx = np.argmax(y_true, axis=1)
    mean_cls_idx = np.argmax(y_mean, axis=1)
    mode_cls_idx = np.argmax(y_mode, axis=1)
    vae_cls_idx  = np.argmax(y_vae, axis=1)
    
    # Label Classi (1-7)
    true_cls = true_cls_idx + 1
    mean_cls = mean_cls_idx + 1
    mode_cls = mode_cls_idx + 1
    vae_cls  = vae_cls_idx + 1
    
    # Livelli di Danno Scalari
    true_levels = y_true[idx_range, true_cls_idx]
    pred_levels_cvae = y_vae[idx_range, vae_cls_idx]
    pred_levels_mean_selected = y_mean[idx_range, mean_cls_idx]
    pred_levels_mode_selected = y_mode[idx_range, mode_cls_idx]
    vae_levels = y_vae[idx_range, true_cls_idx] # Per i plot vecchi

    # Metriche per singolo campione (MAE vettoriale)
    mae_sample_mean = np.mean(np.abs(y_true - y_mean), axis=1)
    mae_sample_mode = np.mean(np.abs(y_true - y_mode), axis=1)
    
    # --- INCERTEZZA ---
    avg_uncertainty = np.mean(y_std, axis=1)
    pred_uncertainty_mode = y_std[idx_range, mode_cls_idx]

    mask_ok_mean = (true_cls == mean_cls)
    mask_ok_mode = (true_cls == mode_cls)

    # ------------------------------------------
    # C. CALCOLO METRICHE COMPLETE
    # ------------------------------------------
    # Accuratezza Classificazione
    acc_cvae = accuracy_score(true_cls, vae_cls)
    acc_mean = accuracy_score(true_cls, mean_cls)
    acc_mode = accuracy_score(true_cls, mode_cls)

    # Metriche Vettoriali
    met_cvae = get_metrics(y_true, y_vae)
    met_mean = get_metrics(y_true, y_mean)
    met_mode = get_metrics(y_true, y_mode)
    
    mae_vec_cvae = met_cvae['MAE']
    mae_vec_mean = met_mean['MAE']
    mae_vec_mode = met_mode['MAE']
    
    cos_sim_cvae = met_cvae['CosSim']
    cos_sim_mean = met_mean['CosSim']
    cos_sim_mode = met_mode['CosSim']

    mse_vec_cvae = mean_squared_error(y_true, y_vae)
    mse_vec_mean = mean_squared_error(y_true, y_mean)
    mse_vec_mode = mean_squared_error(y_true, y_mode)
    
    smape_vec_cvae = smape(y_true, y_vae)
    smape_vec_mean = smape(y_true, y_mean)
    smape_vec_mode = smape(y_true, y_mode)

    # Metriche Scalari (Sull'intensità del danno)
    mae_scalar_level_cvae = mean_absolute_error(true_levels, pred_levels_cvae)
    mae_scalar_level_mean = mean_absolute_error(true_levels, pred_levels_mean_selected)
    mae_scalar_level_mode = mean_absolute_error(true_levels, pred_levels_mode_selected)

    r2_reg_cvae = r2_score(true_levels, pred_levels_cvae)
    r2_reg_mean = r2_score(true_levels, pred_levels_mean_selected)
    r2_reg_mode = r2_score(true_levels, pred_levels_mode_selected)

    # --- 4. Statistiche per il Top 20% dei campioni più incerti ---
    unc_threshold = np.percentile(avg_uncertainty, 80)
    top20_mask = avg_uncertainty >= unc_threshold
    num_top20 = np.sum(top20_mask)
    
    acc_top20_mean = accuracy_score(true_cls[top20_mask], mean_cls[top20_mask])
    acc_top20_mode = accuracy_score(true_cls[top20_mask], mode_cls[top20_mask])
    
    mae_top20_vec_mean = np.mean(mae_sample_mean[top20_mask])
    mae_top20_vec_mode = np.mean(mae_sample_mode[top20_mask])
    
    mae_top20_int_mean = mean_absolute_error(true_levels[top20_mask], pred_levels_mean_selected[top20_mask])
    mae_top20_int_mode = mean_absolute_error(true_levels[top20_mask], pred_levels_mode_selected[top20_mask])

    print("\n" + "="*85)
    print(f"{'METRICA':<20} | {'CVAE (Baseline)':<20} | {'ADVI MEAN Est':<15} | {'ADVI MODE Est':<15}")
    print("-" * 85)
    print(f"{'Accuracy':<20} | {acc_cvae:<20.2%} | {acc_mean:<15.2%} | {acc_mode:<15.2%}")
    print(f"{'MAE (Vector)':<20} | {mae_vec_cvae:<20.5f} | {mae_vec_mean:<15.5f} | {mae_vec_mode:<15.5f}")
    print(f"{'R2 (Regression)':<20} | {r2_reg_cvae:<20.5f} | {r2_reg_mean:<15.5f} | {r2_reg_mode:<15.5f}")
    print("-" * 85)

    # ==========================================
    # D. GENERAZIONE GRAFICI (11 PLOT)
    # ==========================================

    # --- PLOT 1: Confusion Matrix (MEDIA) ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(true_cls, mean_cls, labels=range(1, NUM_CLASSES + 1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=range(1, NUM_CLASSES + 1), yticklabels=range(1, NUM_CLASSES + 1))
    plt.title(f'Confusion Matrix (ADVI Mean)')
    plt.xlabel('Predicted Class (Mean)')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_01_confusion_matrix_MEAN.png"))
    plt.close()

    # --- PLOT 2: Regression (MODA) ---
    plt.figure(figsize=(9, 8))
    plt.scatter(true_levels[mask_ok_mode], pred_levels_mode_selected[mask_ok_mode], 
                alpha=0.6, c='dodgerblue', edgecolor='k', s=70, label='Correct Class')
    plt.scatter(true_levels[~mask_ok_mode], pred_levels_mode_selected[~mask_ok_mode], 
                alpha=0.8, c='crimson', marker='X', s=90, label='Wrong Class')
    plt.plot([0, 0.6], [0, 0.6], 'k--', lw=2, label='Ideal')
    plt.title(f'Regression: True Level vs Selected Class Level (MODE)\nR2: {r2_reg_mode:.3f}')
    plt.xlabel('True Damage Level'); plt.ylabel('Predicted Level (Mode)')
    plt.legend(); plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_02_regression_mode.png"))
    plt.close()

    # --- PLOT 3: Regression (MEDIA) ---
    plt.figure(figsize=(9, 8))
    plt.scatter(true_levels[mask_ok_mean], pred_levels_mean_selected[mask_ok_mean], 
                alpha=0.6, c='mediumseagreen', edgecolor='k', s=70, label='Correct Class')
    plt.scatter(true_levels[~mask_ok_mean], pred_levels_mean_selected[~mask_ok_mean], 
                alpha=0.8, c='crimson', marker='X', s=90, label='Wrong Class')
    plt.plot([0, 0.6], [0, 0.6], 'k--', lw=2, label='Ideal')
    plt.title(f'Regression: True Level vs Selected Class Level (MEAN)\nR2: {r2_reg_mean:.3f}')
    plt.xlabel('True Damage Level'); plt.ylabel('Predicted Level (Mean)')
    plt.legend(); plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_03_regression_mean.png"))
    plt.close()

    # --- PLOT 4: Confronto Errori (MAE) ---
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
    plt.savefig(os.path.join(RESULTS_DIR, "plot_04_error_comparison.png"))
    plt.close()

    # --- PLOT 5: VAE vs ADVI Shift (MEAN) ---
    plt.figure(figsize=(10, 8))
    plt.scatter(true_levels, vae_levels, marker='x', c='gray', s=50, label='VAE Start', alpha=0.5)
    plt.scatter(true_levels, pred_levels_mean_selected, marker='o', c='mediumseagreen', s=60, 
                label='ADVI End (Mean)', edgecolor='k')
    for i in range(len(true_levels)):
        if abs(pred_levels_mean_selected[i] - vae_levels[i]) > 0.015:
            plt.plot([true_levels[i], true_levels[i]], 
                     [vae_levels[i], pred_levels_mean_selected[i]], 
                     c='gray', alpha=0.2)
    plt.plot([0, 0.6], [0, 0.6], 'k--', lw=2)
    plt.title('Refinement Impact: VAE vs ADVI (MEAN)')
    plt.xlabel('True Damage Level'); plt.ylabel('Estimated Level (Mean)')
    plt.legend(); plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_05_vae_shift_MEAN.png"))
    plt.close()

    # --- PLOT 6: Error vs Uncertainty (AVERAGE) ---
    plt.figure(figsize=(10, 8))
    if len(avg_uncertainty) > 1:
        corr, _ = pearsonr(avg_uncertainty, mae_sample_mode)
    else: corr = 0
    
    plt.scatter(avg_uncertainty[mask_ok_mode], mae_sample_mode[mask_ok_mode], 
                c='mediumseagreen', alpha=0.6, label='Correct Class', edgecolor='k', s=60)
    plt.scatter(avg_uncertainty[~mask_ok_mode], mae_sample_mode[~mask_ok_mode], 
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
    plt.savefig(os.path.join(RESULTS_DIR, "plot_06_uncertainty_MEAN_vs_error.png"))
    plt.close()

    # --- PLOT 7: Boxplot Incertezza MEDIA (Correct vs Wrong) ---
    plt.figure(figsize=(8, 6))
    data_to_plot = [avg_uncertainty[mask_ok_mode], avg_uncertainty[~mask_ok_mode]]
    if len(avg_uncertainty[~mask_ok_mode]) > 0:
        bplot = plt.boxplot(data_to_plot, patch_artist=True, tick_labels=['Correct', 'Wrong'])
        colors = ['mediumseagreen', 'crimson']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
    else: plt.text(0.5, 0.5, "No Wrong Predictions!", ha='center', fontsize=14)
    
    plt.title('Correct vs Wrong: AVERAGE Uncertainty Distribution', fontsize=14)
    plt.ylabel('Average Uncertainty', fontsize=12)
    plt.grid(True, axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_07_uncertainty_MEAN_boxplot.png"))
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

    colors_palette = sns.color_palette("viridis", NUM_CLASSES)
    bars = plt.bar(class_labels, avg_unc_per_class, color=colors_palette, alpha=0.8, edgecolor='k')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}\n(n={count})',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.title('Average Uncertainty per Predicted Class', fontsize=14)
    plt.xlabel('Predicted Class'); plt.ylabel('Avg Uncertainty')
    plt.xticks(class_labels); plt.grid(True, axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_08_uncertainty_per_class.png"))
    plt.close()

    # --- PLOT 9: Error vs Uncertainty SPECIFICA ---
    plt.figure(figsize=(10, 8))
    if len(pred_uncertainty_mode) > 1:
        corr_pred, _ = pearsonr(pred_uncertainty_mode, mae_sample_mode)
    else: corr_pred = 0
    
    plt.scatter(pred_uncertainty_mode[mask_ok_mode], mae_sample_mode[mask_ok_mode], 
                c='dodgerblue', alpha=0.6, label='Correct Class', edgecolor='k', s=60)
    plt.scatter(pred_uncertainty_mode[~mask_ok_mode], mae_sample_mode[~mask_ok_mode], 
                c='orange', alpha=0.8, label='Wrong Class', marker='X', edgecolor='k', s=90)
    
    if len(pred_uncertainty_mode) > 1:
        z = np.polyfit(pred_uncertainty_mode, mae_sample_mode, 1)
        p = np.poly1d(z)
        plt.plot(pred_uncertainty_mode, p(pred_uncertainty_mode), "k--", alpha=0.6, lw=2, label=f'Trend (Corr: {corr_pred:.2f})')

    plt.title('Calibration: Error vs PREDICTED CLASS Uncertainty', fontsize=14)
    plt.xlabel('Uncertainty of Predicted Class (Std Dev)', fontsize=12)
    plt.ylabel('Actual Error (MAE Mode)', fontsize=12)
    plt.legend(loc='upper left'); plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_09_uncertainty_PRED_vs_error.png"))
    plt.close()

    # --- PLOT 10: Boxplot Incertezza SPECIFICA ---
    plt.figure(figsize=(8, 6))
    data_to_plot_pred = [pred_uncertainty_mode[mask_ok_mode], pred_uncertainty_mode[~mask_ok_mode]]
    
    if len(pred_uncertainty_mode[~mask_ok_mode]) > 0:
        bplot = plt.boxplot(data_to_plot_pred, patch_artist=True, tick_labels=['Correct', 'Wrong'])
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

    # --- PLOT 11: MAE per Classe (SOLO MEAN) ---
    plt.figure(figsize=(10, 6))
    mae_per_class_mean = []
    unc_per_class_mean = []
    class_labels = range(1, NUM_CLASSES + 1)
    
    for c in class_labels:
        mask_c = (true_cls == c)
        if np.sum(mask_c) > 0:
            mae_per_class_mean.append(np.mean(mae_sample_mean[mask_c]))
            unc_per_class_mean.append(np.mean(y_std[mask_c]))
        else:
            mae_per_class_mean.append(0.0)
            unc_per_class_mean.append(0.0)

    plt.bar(class_labels, mae_per_class_mean, color='mediumseagreen', alpha=0.8, edgecolor='k', width=0.6)
    plt.title('Mean Absolute Error (MAE) per True Class (Mean Estimator)', fontsize=14)
    plt.xlabel('True Class Index'); plt.ylabel('Average MAE')
    plt.xticks(class_labels); plt.grid(True, axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_11_mae_per_class_MEAN_ONLY.png"))
    plt.close()

    print(f"\n[DONE] Generati 11 grafici in: {RESULTS_DIR}")

    # ==========================================
    # E. SCRITTURA FILE DI RIEPILOGO TESTUALE
    # ==========================================
    SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary_statistics.txt")
    
    # Calcoli per Refinement
    improved_mask = (vae_cls != true_cls) & (mode_cls == true_cls)
    n_improved = np.sum(improved_mask)
    degraded_mask = (vae_cls == true_cls) & (mode_cls != true_cls)
    n_degraded = np.sum(degraded_mask)

    global_avg_uncertainty = np.mean(y_std)
    avg_time = df['Time'].mean() if 'Time' in df.columns else 0.0

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write("===================================================================================================\n")
            f.write("                                RIEPILOGO STATISTICO COMPLETO ADVI\n")
            f.write("===================================================================================================\n\n")
            
            f.write(f"Timestamp: {pd.Timestamp.now()}\n")
            f.write(f"Campioni: {len(df)}\n")
            f.write(f"Tempo Medio: {avg_time:.4f} s\n")
            f.write(f"Incertezza Globale (Avg Std): {global_avg_uncertainty:.6f}\n\n")
            
            # --- TABELLA A 3 COLONNE ---
            f.write("-" * 105 + "\n")
            f.write(f"{'METRICA':<20} | {'CVAE (Baseline)':<25} | {'ADVI MEAN Est':<25} | {'ADVI MODE Est':<25}\n")
            f.write("-" * 105 + "\n")
            f.write(f"{'MAE (Vector)':<20} | {mae_vec_cvae:<25.6f} | {mae_vec_mean:<25.6f} | {mae_vec_mode:<25.6f}\n")
            f.write(f"{'MAE (Intensity)':<20} | {mae_scalar_level_cvae:<25.6f} | {mae_scalar_level_mean:<25.6f} | {mae_scalar_level_mode:<25.6f}\n")
            f.write(f"{'MSE (Vector)':<20} | {mse_vec_cvae:<25.6f} | {mse_vec_mean:<25.6f} | {mse_vec_mode:<25.6f}\n")
            f.write(f"{'RMSE (Vector)':<20} | {np.sqrt(mse_vec_cvae):<25.6f} | {np.sqrt(mse_vec_mean):<25.6f} | {np.sqrt(mse_vec_mode):<25.6f}\n")
            f.write(f"{'R2 (Regression)':<20} | {r2_reg_cvae:<25.6f} | {r2_reg_mean:<25.6f} | {r2_reg_mode:<25.6f}\n")
            f.write(f"{'Cos Sim':<20} | {cos_sim_cvae:<25.6f} | {cos_sim_mean:<25.6f} | {cos_sim_mode:<25.6f}\n")
            f.write(f"{'SMAPE':<20} | {smape_vec_cvae:<25.6f} | {smape_vec_mean:<25.6f} | {smape_vec_mode:<25.6f}\n")
            f.write("-" * 105 + "\n\n")
            
            f.write("--- CLASSIFICAZIONE ---\n")
            f.write(f"Acc (CVAE): {acc_cvae:.2%} ({int(acc_cvae*len(df))}/{len(df)})\n")
            f.write(f"Acc (Mean): {acc_mean:.2%} ({int(acc_mean*len(df))}/{len(df)})\n")
            f.write(f"Acc (Mode): {acc_mode:.2%} ({int(acc_mode*len(df))}/{len(df)})\n\n")
            
            f.write("--- DETTAGLIO PER CLASSE REALE (Mean Est) ---\n")
            f.write(f"{'Class':<6} | {'Avg MAE':<12} | {'Avg Unc (Std)':<22} | {'Count':<6}\n")
            f.write("-" * 60 + "\n")
            
            # Calcolo incertezze per classe (Usando la logica esistente di ADVI)
            for i, c in enumerate(class_labels):
                count_c = np.sum(true_cls == c)
                unc_c = np.mean(unc_per_class_mean[i])
                f.write(f"{c:<6} | {mae_per_class_mean[i]:.6f}      | {unc_c:.6f}                 | {count_c:<6}\n")
            f.write("\n")

            # --- NUOVA SEZIONE: TOP 20% INCERTI ---
            f.write("--- ANALISI DEL 20% DEI CAMPIONI PIÙ INCERTI (Top 20% Avg Std) ---\n")
            f.write(f"Soglia Incertezza (80° percentile): {unc_threshold:.6f}\n")
            f.write(f"Numero di campioni valutati: {num_top20}\n\n")
            
            f.write(f"{'METRICA':<20} | {'MEAN Estimator':<20} | {'MODE Estimator':<20}\n")
            f.write("-" * 65 + "\n")
            f.write(f"{'Accuracy':<20} | {acc_top20_mean:<20.2%} | {acc_top20_mode:<20.2%}\n")
            f.write(f"{'MAE (Vector)':<20} | {mae_top20_vec_mean:<20.6f} | {mae_top20_vec_mode:<20.6f}\n")
            f.write(f"{'MAE (Intensity)':<20} | {mae_top20_int_mean:<20.6f} | {mae_top20_int_mode:<20.6f}\n")
            f.write("-" * 65 + "\n\n")

            f.write("--- REFINEMENT (CVAE vs ADVI MODE) ---\n")
            f.write(f"Migliorati (CVAE err -> ADVI ok): {n_improved}\n")
            f.write(f"Peggiorati (CVAE ok  -> ADVI err): {n_degraded}\n")
            f.write(f"Netto: {n_improved - n_degraded:+d}\n\n")

            f.write("--- ERRORI CLASSIFICAZIONE (MODE) ---\n")
            mask_wrong = (true_cls != mode_cls)
            if np.sum(mask_wrong) > 0:
                f.write(f"Totale Errori: {np.sum(mask_wrong)}\n")
                f.write(f"{'Index':<8} | {'True':<6} | {'Pred':<6} | {'MAE (Mode)':<12} | {'Uncertainty':<24}\n")
                f.write("-" * 75 + "\n")
                
                idxs = df.index[mask_wrong]
                t_wrong = true_cls[mask_wrong]
                p_wrong = mode_cls[mask_wrong]
                mae_wrong = mae_sample_mode[mask_wrong]
                unc_wrong_spec = pred_uncertainty_mode[mask_wrong]
                
                for idx_v, t, p, m, u in zip(idxs, t_wrong, p_wrong, mae_wrong, unc_wrong_spec):
                     f.write(f"{df.iloc[idx_v]['Index']:<8} | {t:<6} | {p:<6} | {m:.6f}       | {u:.6f}\n")
                
                f.write(f"\n{'MAE (Intensity)':<20} | {mae_scalar_level_cvae:<25.6f} | {mae_scalar_level_mean:<25.6f} | {mae_scalar_level_mode:<25.6f}\n")

            else:
                f.write("Nessun errore.\n")

    print(f"Statistiche salvate in: {SUMMARY_PATH}")