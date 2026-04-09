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

# Cartella di output
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_svgd_bridge_stats3") 

# Nome file CSV
CSV_FILENAME = "results_svgd_bridge_final.csv"
CSV_PATH = os.path.join(RESULTS_DIR, CSV_FILENAME)

# Numero classi
NUM_CLASSES = 6

# ==========================================
# 2. UTILS DI PARSING
# ==========================================
def parse_vector_col(df, col_name):
    """
    Converte le stringhe del CSV in array numpy.
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
    """Calcola metriche standard per vettori."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "CosSim": np.mean(cosine_similarity(y_true, y_pred).diagonal())
    }

def calculate_smape(y_true, y_pred):
    """Calcola SMAPE gestendo la divisione per zero."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_pred) + np.abs(y_true)) / 2 + 1e-10
    return np.mean(numerator / denominator) * 100.0

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    
    if not os.path.exists(CSV_PATH):
        print(f"\n[ERRORE] Il file dati non esiste: {CSV_PATH}")
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
    
    # Indici
    true_cls_idx = np.argmax(y_true, axis=1)
    mean_cls_idx = np.argmax(y_mean, axis=1)
    mode_cls_idx = np.argmax(y_mode, axis=1)

    # Label (1-7)
    true_cls = true_cls_idx + 1
    mean_cls = mean_cls_idx + 1
    mode_cls = mode_cls_idx + 1
    
    # Livelli Scalari
    true_levels = y_true[idx_range, true_cls_idx]
    vae_levels = y_vae[idx_range, true_cls_idx]
    pred_levels_mean_selected = y_mean[idx_range, mean_cls_idx]
    pred_levels_mode_selected = y_mode[idx_range, mode_cls_idx]

    # Metriche Campione (MAE)
    mae_sample_mean = np.mean(np.abs(y_true - y_mean), axis=1)
    mae_sample_mode = np.mean(np.abs(y_true - y_mode), axis=1)
    
    # Incertezza
    avg_uncertainty = np.mean(y_std, axis=1)
    pred_uncertainty_mode = y_std[idx_range, mode_cls_idx]

    mask_ok = (true_cls == mode_cls)

    # ------------------------------------------
    # C. CALCOLO METRICHE COMPLETE
    # ------------------------------------------
    # 1. Metriche Globali Vettoriali
    mae_vec_mean = mean_absolute_error(y_true, y_mean)
    mae_vec_mode = mean_absolute_error(y_true, y_mode)
    mse_vec_mean = mean_squared_error(y_true, y_mean)
    mse_vec_mode = mean_squared_error(y_true, y_mode)
    smape_vec_mean = calculate_smape(y_true, y_mean)
    smape_vec_mode = calculate_smape(y_true, y_mode)
    
    cos_sim_mean = np.mean(np.diag(cosine_similarity(y_true, y_mean)))
    cos_sim_mode = np.mean(np.diag(cosine_similarity(y_true, y_mode)))

    # 2. Metriche Scalari (Regressione R2)
    r2_reg_mean = r2_score(true_levels, pred_levels_mean_selected)
    r2_reg_mode = r2_score(true_levels, pred_levels_mode_selected)

    # Calcolo MAE scalare (solo sul livello di danno nella classe predetta vs classe reale)
    mae_scalar_level_mean = mean_absolute_error(true_levels, pred_levels_mean_selected)
    mae_scalar_level_mode = mean_absolute_error(true_levels, pred_levels_mode_selected)
    
    # 3. Classificazione
    acc_mean = accuracy_score(true_cls, mean_cls)
    acc_mode = accuracy_score(true_cls, mode_cls)

    print("\n" + "="*65)
    print(f"{'METRICA':<15} | {'MEAN Est':<12} | {'MODE Est':<12}")
    print("-" * 65)
    print(f"Accuracy        | {acc_mean:.2%}      | {acc_mode:.2%}")
    print(f"MAE (Vector)    | {mae_vec_mean:.5f}      | {mae_vec_mode:.5f}")
    print(f"R2 (Regression) | {r2_reg_mean:.5f}      | {r2_reg_mode:.5f}")
    print("-" * 65)

    # ==========================================
    # D. GENERAZIONE GRAFICI (12 PLOT)
    # ==========================================

    # 1. Confusion Matrix Mean
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(true_cls, mean_cls, labels=range(1, NUM_CLASSES + 1)), 
                annot=True, fmt='d', cmap='Greens')
    plt.title(f'Confusion Matrix (SVGD Mean)\nAcc: {acc_mean:.2%}')
    plt.xlabel('Predicted Class (Mean)'); plt.ylabel('True Class')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "plot_01_confusion_matrix_MEAN_svgd.png")); plt.close()

    # 1-bis. Confusion Matrix Mode
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(true_cls, mode_cls, labels=range(1, NUM_CLASSES + 1)), 
                annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (SVGD Mode)\nAcc: {acc_mode:.2%}')
    plt.xlabel('Predicted Class (Mode)'); plt.ylabel('True Class')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "plot_01b_confusion_matrix_MODE_svgd.png")); plt.close()

    # 2. Regression Mode
    plt.figure(figsize=(9, 8))
    plt.scatter(true_levels[mask_ok], pred_levels_mode_selected[mask_ok], alpha=0.6, c='dodgerblue', label='Correct')
    plt.scatter(true_levels[~mask_ok], pred_levels_mode_selected[~mask_ok], alpha=0.8, c='crimson', marker='X', label='Wrong')
    plt.plot([0, 0.6], [0, 0.6], 'k--'); plt.legend(); plt.grid(True, ls='--')
    plt.title(f'Regression Mode (SVGD)\nR2: {r2_reg_mode:.3f}')
    plt.savefig(os.path.join(RESULTS_DIR, "plot_02_regression_mode_svgd.png")); plt.close()

    # 3. Regression Mean
    plt.figure(figsize=(9, 8))
    mask_ok_mean = (true_cls == mean_cls)
    plt.scatter(true_levels[mask_ok_mean], pred_levels_mean_selected[mask_ok_mean], alpha=0.6, c='mediumseagreen', label='Correct')
    plt.scatter(true_levels[~mask_ok_mean], pred_levels_mean_selected[~mask_ok_mean], alpha=0.8, c='crimson', marker='X', label='Wrong')
    plt.plot([0, 0.6], [0, 0.6], 'k--'); plt.legend(); plt.grid(True, ls='--')
    plt.title(f'Regression Mean (SVGD)\nR2: {r2_reg_mean:.3f}')
    plt.savefig(os.path.join(RESULTS_DIR, "plot_03_regression_mean_svgd.png")); plt.close()

    # 4. Error Comparison
    plt.figure(figsize=(9, 8))
    plt.scatter(mae_sample_mean, mae_sample_mode, alpha=0.6, c='purple')
    mx = max(mae_sample_mean.max(), mae_sample_mode.max()) * 1.05
    plt.plot([0, mx], [0, mx], 'r--')
    plt.xlabel('MAE Mean'); plt.ylabel('MAE Mode'); plt.title('SVGD Error Comparison')
    plt.savefig(os.path.join(RESULTS_DIR, "plot_04_error_comparison_svgd.png")); plt.close()

    # 5. VAE Shift
    plt.figure(figsize=(10, 8))
    plt.scatter(true_levels, vae_levels, marker='x', c='gray', alpha=0.5, label='VAE')
    plt.scatter(true_levels, pred_levels_mean_selected, c='green', alpha=0.5, label='SVGD Mean')
    plt.plot([0, 0.6], [0, 0.6], 'k--'); plt.legend()
    plt.title('Refinement: VAE vs SVGD')
    plt.savefig(os.path.join(RESULTS_DIR, "plot_05_vae_shift_MEAN_svgd.png")); plt.close()

    # 6. Uncertainty vs Error (Mean)
    plt.figure(figsize=(10, 8))
    if len(avg_uncertainty) > 1:
        corr, _ = pearsonr(avg_uncertainty, mae_sample_mode)
        plt.scatter(avg_uncertainty, mae_sample_mode, alpha=0.5, c='teal')
        z = np.polyfit(avg_uncertainty, mae_sample_mode, 1); p = np.poly1d(z)
        plt.plot(avg_uncertainty, p(avg_uncertainty), "k--", label=f'Corr: {corr:.2f}')
    plt.title('Calibration (Avg Uncertainty)'); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_06_unc_vs_error_svgd.png")); plt.close()

    # 7. Boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot([avg_uncertainty[mask_ok], avg_uncertainty[~mask_ok]], labels=['Correct', 'Wrong'])
    plt.title('Uncertainty Distribution'); plt.grid(axis='y')
    plt.savefig(os.path.join(RESULTS_DIR, "plot_07_unc_boxplot_svgd.png")); plt.close()

    # 8. Unc per Class
    plt.figure(figsize=(10, 6))
    unc_per_cls = [np.mean(avg_uncertainty[mode_cls == c]) if np.sum(mode_cls==c)>0 else 0 for c in range(1,8)]
    plt.bar(range(1,8), unc_per_cls, color=sns.color_palette("viridis", 7)); plt.title('Avg Uncertainty per Pred Class')
    plt.savefig(os.path.join(RESULTS_DIR, "plot_08_unc_per_class_svgd.png")); plt.close()

    # 9. Calibration Specific
    plt.figure(figsize=(10, 8))
    plt.scatter(pred_uncertainty_mode, mae_sample_mode, alpha=0.5, c='orange')
    plt.xlabel('Pred Class Unc'); plt.ylabel('MAE Mode'); plt.title('Calibration Specific')
    plt.savefig(os.path.join(RESULTS_DIR, "plot_09_unc_spec_vs_err_svgd.png")); plt.close()

    # 10. Boxplot Specific
    plt.figure(figsize=(8, 6))
    plt.boxplot([pred_uncertainty_mode[mask_ok], pred_uncertainty_mode[~mask_ok]], labels=['Correct', 'Wrong'])
    plt.title('Specific Uncertainty Distribution'); plt.grid(axis='y')
    plt.savefig(os.path.join(RESULTS_DIR, "plot_10_unc_spec_boxplot_svgd.png")); plt.close()

    # 11. MAE per Class
    plt.figure(figsize=(10, 6))
    mae_per_cls = [np.mean(mae_sample_mean[true_cls == c]) if np.sum(true_cls==c)>0 else 0 for c in range(1,8)]
    colors = sns.color_palette("magma", 7)
    bars = plt.bar(range(1,8), mae_per_cls, color=colors, edgecolor='k', alpha=0.8)
    plt.axhline(mae_vec_mean, color='red', linestyle='--', label=f'Global: {mae_vec_mean:.4f}')
    for bar in bars:
        h = bar.get_height()
        if h>0: plt.text(bar.get_x()+bar.get_width()/2, h, f'{h:.4f}', ha='center', va='bottom')
    plt.title('MAE per True Class (SVGD Mean)'); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_11_mae_per_class_svgd.png")); plt.close()

    print("[DONE] Grafici generati.")

    # ==========================================
    # E. SCRITTURA FILE TESTUALE (Identico a ADVI)
    # ==========================================
    SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary_statistics.txt")
    
    # Dati Refinement
    vae_cls_idx = np.argmax(y_vae, axis=1)
    vae_cls = vae_cls_idx + 1
    improved = np.sum((vae_cls != true_cls) & (mean_cls == true_cls))
    degraded = np.sum((vae_cls == true_cls) & (mean_cls != true_cls))
    
    avg_time = df['Time'].mean() if 'Time' in df.columns else 0.0

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("===================================================================\n")
        f.write("             RIEPILOGO STATISTICO COMPLETO SVGD\n")
        f.write("===================================================================\n\n")
        
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Campioni: {len(df)}\n")
        f.write(f"Tempo Medio: {avg_time:.4f} s\n")
        f.write(f"Incertezza Globale (Avg Std): {np.mean(y_std):.6f}\n\n")
        
        f.write("-" * 85 + "\n")
        f.write(f"{'METRICA':<15} | {'MEAN Estimator':<25} | {'MODE Estimator':<25}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'MAE (Vector)':<15} | {mae_vec_mean:.6f}                  | {mae_vec_mode:.6f}\n")
        f.write(f"{'MAE (Intensity)':<15} | {mae_scalar_level_mean:.6f}                  | {mae_scalar_level_mode:.6f}\n")
        f.write(f"{'MSE (Vector)':<15} | {mse_vec_mean:.6f}                  | {mse_vec_mode:.6f}\n")
        f.write(f"{'RMSE (Vector)':<15} | {np.sqrt(mse_vec_mean):.6f}                  | {np.sqrt(mse_vec_mode):.6f}\n")
        f.write(f"{'R2 (Regression)':<15} | {r2_reg_mean:.6f}                  | {r2_reg_mode:.6f}\n")
        f.write(f"{'Cos Sim':<15} | {cos_sim_mean:.6f}                  | {cos_sim_mode:.6f}\n")
        f.write(f"{'SMAPE':<15} | {smape_vec_mean:.6f}                  | {smape_vec_mode:.6f}\n")
        f.write("-" * 85 + "\n\n")
        
        f.write("--- CLASSIFICAZIONE ---\n")
        f.write(f"Acc (Mean): {acc_mean:.2%} ({int(acc_mean*len(df))}/{len(df)})\n")
        f.write(f"Acc (Mode): {acc_mode:.2%} ({int(acc_mode*len(df))}/{len(df)})\n\n")
        
        f.write("--- DETTAGLIO PER CLASSE REALE (Mean Est) ---\n")
        f.write(f"{'Class':<6} | {'Avg MAE':<12} | {'Avg Unc (Std)':<22} | {'Count':<6}\n")
        f.write("-" * 60 + "\n")
        
        # Calcolo incertezze per classe
        unc_per_class = []
        for c in range(1, NUM_CLASSES + 1):
            mask = (true_cls == c)
            if np.sum(mask) > 0:
                unc_c = np.mean(avg_uncertainty[mask])
            else: unc_c = 0
            unc_per_class.append(unc_c)

        for i, c in enumerate(range(1, NUM_CLASSES + 1)):
            count_c = np.sum(true_cls == c)
            f.write(f"{c:<6} | {mae_per_cls[i]:.6f}      | {unc_per_class[i]:.6f}                 | {count_c:<6}\n")
        f.write("\n")

        f.write("--- REFINEMENT (CVAE vs SVGD MODE) ---\n")
        f.write(f"Migliorati (CVAE err -> SVGD ok): {improved}\n")
        f.write(f"Peggiorati (CVAE ok  -> SVGD err): {degraded}\n")
        f.write(f"Netto: {improved - degraded:+d}\n\n")

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
        else:
            f.write("Nessun errore.\n")

    print(f"Statistiche salvate in: {SUMMARY_PATH}")